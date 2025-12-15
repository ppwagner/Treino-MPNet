
import os
import json
from dataclasses import dataclass
import time
import math

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from mpnet_masks import make_unified_square_mask

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

@dataclass 
class DatasetArgs:
        dataset_dir: str
        batch_size: int
        seq_len: int
        grad_accum_steps: int
        val_tokens: int
        tokens_per_batch: int
        val_tokens_padding: int
        num_processes: int
    

class DistributedShardedDataset(IterableDataset):
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, dataset_dir, batch_size, seq_len, process_rank, num_processes, 
                 grad_accum_steps, val_tokens=0, val_tokens_padding=10**6, mask_amount=0.15, mask_token_id=103, whole_word_mask=False):
        self.dataset_dir            = dataset_dir
        self.process_rank           = process_rank
        self.num_processes          = num_processes
        self.batch_size             = batch_size
        self.seq_len                = seq_len
        self.tokens_per_world_fwd   = batch_size * seq_len * num_processes
        self.tokens_per_batch       = batch_size * seq_len * num_processes * grad_accum_steps
        self.tokens_per_fwd         = batch_size * seq_len
        self.grad_accum_steps       = grad_accum_steps
        self.val_tokens             = val_tokens


        self.val_tokens_padding = val_tokens_padding if self.val_tokens > 0 else 0
        self.mask_amount = mask_amount
        self.mask_token_id = mask_token_id
        self.whole_word_mask = whole_word_mask
        self.tokenizer = None
        self.tokenizer_name = None
        self.pad_token_id = 1 # Default pad token id
        
        print0(f'Valiation tokens:                              {self.val_tokens:16,}')
        print0(f'Valiation tokens padding:                      {self.val_tokens_padding:16,}')


        # glob files that match the pattern
        self.dataset_path = os.path.join('data', dataset_dir)

        # kick things off
        self.current_shard      = None
        self.global_position    = None
        self.world_position     = None
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard   = 0
            self.global_position = 0
            self.input_ids, self.seq_codes = self.load_data_shard(self.current_shard)
        self.world_position    = self.val_tokens_padding + self.val_tokens
        self.current_position  = self.process_rank * self.batch_size * self.seq_len
        self.current_position += self.val_tokens_padding + self.val_tokens
    
    def load_data_shard(self, idx):
        filepath = os.path.join(self.dataset_path, f"sample_{idx:06d}.pt")
        dataset = torch.load(filepath)
        
        if self.whole_word_mask:
            tokenizer_name = dataset.get('tokenizer')
            if tokenizer_name and tokenizer_name != self.tokenizer_name:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    self.tokenizer_name = tokenizer_name
                    print0(f"Loaded tokenizer: {tokenizer_name}")
                except Exception as e:
                    print0(f"Failed to load tokenizer {tokenizer_name}: {e}")
                    # Fallback? Should probably fail if WWP is requested.
                    
        return dataset['input_ids'], dataset['seq_codes']
    
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        # if loading the next batch would be out of bounds load shards until we have enough data
        self.world_position += self.tokens_per_world_fwd*self.grad_accum_steps
        while self.world_position + 1 > len(self.input_ids):
            self.load_next_shard()

        batches = []
        for i in range(self.grad_accum_steps):
            # # if loading the next batch would be out of bounds load shards until we have enough data
            # while self.current_position + self.tokens_per_fwd + 1 > len(self.input_ids):
            #     self.load_next_shard()
            batches.append(self.get_batch(self.current_position, self.current_position+self.tokens_per_fwd))
            # advance the start pointer in current shard
            self.current_position += self.tokens_per_world_fwd
        return batches
    
    def load_next_shard(self):
        self.current_shard = (self.current_shard + 1)
        if os.path.exists(os.path.join(self.dataset_path, f"sample_{self.current_shard:06d}.pt")) is False:
            print(f'Rank {self.process_rank} has finished the dataset')
            raise StopIteration
        self.current_position = self.process_rank * self.tokens_per_fwd
        self.world_position   = self.tokens_per_batch

        remainder_pos = round_to_multiple(len(self.input_ids), multiple=self.tokens_per_world_fwd*self.grad_accum_steps, up=False)
        self.input_ids = self.input_ids[remainder_pos:]
        self.seq_codes = self.seq_codes[remainder_pos:]

        new_input_ids, new_seq_codes = self.load_data_shard(self.current_shard)
        self.input_ids = torch.cat([self.input_ids, new_input_ids], dim=0)
        self.seq_codes = torch.cat([self.seq_codes, new_seq_codes], dim=0)

        self.global_position += new_input_ids.size(0)

    def get_val_dataset(self, device='cpu'):
        if self.val_tokens == 0:
            return None
        else:
            local_val_batches_count = self.val_tokens // self.tokens_per_world_fwd
            # val_batches = torch.empty((local_val_batches_count, 3, self.batch_size, self.seq_len), dtype=torch.int64)
            val_batches = []
            pos = self.process_rank * self.tokens_per_fwd
            for i in range(local_val_batches_count):
                batch_items = self.get_batch(pos, pos+self.batch_size*self.seq_len)
                val_batches.append(batch_items)
                pos += self.tokens_per_world_fwd
            return val_batches
    

    def _process_segment(self, segment, seg_code):
        seg_len = len(segment)
        if seg_len == 0:
            return None

        # 1. Output Targets and Pos
        target = torch.full((seg_len,), -1, dtype=torch.long)
        pos = torch.arange(seg_len, dtype=torch.long)
        
        # 2. Determine mask amount
        if isinstance(self.mask_amount, float):
            n_mask = max(1, int(seg_len * self.mask_amount))
        else:
            n_mask = min(seg_len, int(self.mask_amount))
        
        # Enforce mpnet constraint: seg_len >= 2 * n_mask for make_unified_square_mask
        if n_mask * 2 > seg_len:
            n_mask = seg_len // 2
            
        if n_mask == 0:
            return None
        
        # 3. Permute
        if self.whole_word_mask and self.tokenizer is not None:
             # Convert to list for tokenization
            tokens = self.tokenizer.convert_ids_to_tokens(segment.tolist())
            
            # Group indices by word
            word_groups = []
            current_group = []
            
            for i, token in enumerate(tokens):
                # Check for start of word. Mistral/Llama usually use ' ' (U+2581) to denote space preceding word.
                # If token starts with space substitute, it's a new word.
                # Also index 0 is always new word (or part of first word).
                # Note: ' ' is the SPIECE_UNDERLINE for SentencePiece (\u2581)
                # Ġ is for Byte-BPE (GPT2/RoBERTa)
                is_start = token.startswith(' ') or token.startswith(' ') or token.startswith('Ġ') or token.startswith('\u2581')
                
                if is_start and current_group:
                    word_groups.append(current_group)
                    current_group = []
                
                current_group.append(i)
            
            if current_group:
                word_groups.append(current_group)
                
            # Permute words
            n_words = len(word_groups)
            word_perm = torch.randperm(n_words)
            
            # Flatten to get token permutation
            perm_indices = []
            for i in word_perm:
                perm_indices.extend(word_groups[i.item()])
            
            perm = torch.tensor(perm_indices, dtype=torch.long, device=segment.device)
            
        else:
            perm = torch.randperm(seg_len)
            
        segment_permuted = segment[perm]
        pos_perm = pos[perm]
        
        # 4. Create Extra Mask Tokens
        extra_tokens = torch.full((n_mask,), self.mask_token_id, dtype=torch.long)
        extra_pos = pos_perm[-n_mask:] 
        extra_targets = segment_permuted[-n_mask:]
        
        # 5. Create Seq Codes
        extra_seq_codes = torch.full((n_mask,), seg_code, dtype=torch.long)
        segment_seq_codes = torch.full((len(segment_permuted),), seg_code, dtype=torch.long)
        
        # 6. Concatenate
        new_seg_input = torch.cat([segment_permuted, extra_tokens, extra_tokens])
        new_seg_pos = torch.cat([pos_perm, extra_pos, extra_pos])
        new_seg_seq_codes = torch.cat([segment_seq_codes, extra_seq_codes, extra_seq_codes])
        
        # Target: segment_permuted for context (ignored by mask), extra_targets for prediction
        new_seg_target = torch.cat([segment_permuted, extra_targets, extra_targets])
        
        # Loss Mask: 0 for context, 1 for prediction
        context_loss_mask = torch.zeros((len(segment_permuted)+n_mask,), dtype=torch.bool)
        extra_loss_mask = torch.ones((n_mask,), dtype=torch.bool)
        new_seg_loss_mask = torch.cat([context_loss_mask, extra_loss_mask])
        
        # 7. Generate Mask
        # a = seg_len - n_mask, b = n_mask
        a = seg_len - n_mask
        a = seg_len 
        b = n_mask
        # Note: tensor arg is unused in make_unified_square_mask currently but we pass a dummy
        mask = make_unified_square_mask(new_seg_input, a, b, kind='MPLM') 
        # mask shape: (a + 2*b, a + 2*b) = (seg_len + n_mask, seg_len + n_mask)
        # Because seg_len = a + b.
        
        return new_seg_input, new_seg_pos, new_seg_seq_codes, new_seg_target, mask, new_seg_loss_mask

    def get_batch(self, start, end):
        input_ids = self.input_ids[start:end].long()
        seq_codes = self.seq_codes[start:end].long()

        input_tensor = input_ids.view(self.batch_size, self.seq_len) 
        seq_codes_tensor = seq_codes.view(self.batch_size, self.seq_len)
        
        # Lists to hold processed rows
        out_input_ids = []
        out_pos_ids = []
        out_seq_codes = []
        out_targets = []
        out_masks = []
        out_loss_masks = []
        
        max_len = 0

        for b in range(self.batch_size):
            row_input = input_tensor[b]
            row_code = seq_codes_tensor[b]
            
            # Identify splits where seq_code changes
            transitions = (row_code[1:] != row_code[:-1])
            split_indices = torch.where(transitions)[0] + 1
            split_indices = torch.cat([torch.tensor([0], device=split_indices.device), 
                                       split_indices,
                                       torch.tensor([len(row_code)], device=split_indices.device)])
            
            row_segments_input = []
            row_segments_pos = []
            row_segments_seq = []
            row_segments_target = []
            row_segments_mask = []
            row_segments_loss_mask = []
            
            for i in range(len(split_indices) - 1):
                s_start = split_indices[i]
                s_end = split_indices[i+1]
                
                segment = row_input[s_start:s_end]
                seg_code = row_code[s_start]
                
                result = self._process_segment(segment, seg_code)
                if result is None: continue
                
                s_input, s_pos, s_seq, s_target, s_mask, s_loss_mask = result
                
                row_segments_input.append(s_input)
                row_segments_pos.append(s_pos)
                row_segments_seq.append(s_seq)
                row_segments_target.append(s_target)
                row_segments_mask.append(s_mask)
                row_segments_loss_mask.append(s_loss_mask)

            # Recombine row segments
            if len(row_segments_input) > 0:
                row_combined_input = torch.cat(row_segments_input)
                row_combined_pos = torch.cat(row_segments_pos)
                row_combined_seq = torch.cat(row_segments_seq)
                row_combined_target = torch.cat(row_segments_target)
                # Combine masks block-diagonally
                row_combined_mask = torch.block_diag(*row_segments_mask)
                row_combined_loss_mask = torch.cat(row_segments_loss_mask)
            else:
                # Fallback for empty row
                row_combined_input = row_input 
                row_combined_pos = torch.arange(len(row_input))
                row_combined_seq = row_code
                row_combined_target = row_input.clone() # Use input as target (though likely ignored)
                # Full attention for fallback? Or empty?
                row_combined_mask = torch.ones((len(row_input), len(row_input)), dtype=torch.bool)
                row_combined_loss_mask = torch.zeros((len(row_input),), dtype=torch.bool) # No loss

            out_input_ids.append(row_combined_input)
            out_pos_ids.append(row_combined_pos)
            out_seq_codes.append(row_combined_seq)
            out_targets.append(row_combined_target)
            out_masks.append(row_combined_mask)
            out_loss_masks.append(row_combined_loss_mask)
            
            max_len = max(max_len, len(row_combined_input))

        # Pad to max_len
        final_input = torch.full((self.batch_size, max_len), self.pad_token_id, dtype=torch.long)
        
        final_pos = torch.zeros((self.batch_size, max_len), dtype=torch.long)
        final_seq = torch.zeros((self.batch_size, max_len), dtype=torch.long)
        final_target = torch.full((self.batch_size, max_len), self.pad_token_id, dtype=torch.long)
        final_mask = torch.zeros((self.batch_size, max_len, max_len), dtype=torch.bool) # False = Blocked
        final_loss_mask = torch.zeros((self.batch_size, max_len), dtype=torch.bool)
        
        for b in range(self.batch_size):
            L = len(out_input_ids[b])
            final_input[b, :L] = out_input_ids[b]
            final_pos[b, :L] = out_pos_ids[b]
            final_seq[b, :L] = out_seq_codes[b]
            final_target[b, :L] = out_targets[b]
            
            # Mask assumes L x L block is valid
            mask_h, mask_w = out_masks[b].shape
            final_mask[b, :mask_h, :mask_w] = out_masks[b]
            final_loss_mask[b, :L] = out_loss_masks[b]
            
        return final_input, final_pos, final_seq, final_target, final_mask, final_loss_mask
    
    def get_args(self):
        args = DatasetArgs(
            dataset_dir=self.dataset_dir,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            grad_accum_steps=self.grad_accum_steps,
            val_tokens=self.val_tokens,
            tokens_per_batch=self.tokens_per_batch,
            val_tokens_padding=self.val_tokens_padding,
            num_processes=self.num_processes,
        )
        return args


# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def round_to_multiple(x, multiple=1, up=False):
    if x % multiple == 0:
        return x
    return x + up*multiple - (x%multiple)


# learning rate decay scheduler (cosine with warmup)
def set_lr(optimizer, it, num_iterations, args):
    for param_group in optimizer.param_groups:
        min_lr = param_group['init_lr'] * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            param_group['lr'] = param_group['init_lr'] * (it+1) / args.warmup_iters
            continue
        # 2) if it > lr_decay_iters, return min learning rate
        if it > num_iterations:
            param_group['lr'] = min_lr
            continue
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        param_group['lr'] = min_lr + coeff * (param_group['init_lr'] - min_lr)
    return optimizer

def compute_radam_lr(radam_optimizer):
    def _compute_rect(rho_t, rho_inf):
        return (
            (rho_t - 4)
            * (rho_t - 2)
            * rho_inf
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        ) ** 0.5


    step = radam_optimizer.state[radam_optimizer.param_groups[0]['params'][0]]['step'].item()
    beta2 = radam_optimizer.param_groups[0]['betas'][1]
    bias_correction2 = 1 - beta2**step

    # maximum length of the approximated SMA
    rho_inf = 2 / (1 - beta2) - 1
    # compute the length of the approximated SMA
    rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

    if rho_t > 5:
        return _compute_rect(rho_t, rho_inf)*radam_optimizer.param_groups[0]['lr']
    else:
        return None



# # learning rate decay scheduler (cosine with warmup)
# def get_lr(it):
#     min_lr = args.learning_rate * args.learning_rate_decay_frac
#     # 1) linear warmup for warmup_iters steps
#     if it < args.warmup_iters:
#         return args.learning_rate * (it+1) / args.warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > num_iterations:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - args.warmup_iters) / (num_iterations - args.warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
#     return min_lr + coeff * (args.learning_rate - min_lr)
class StateMonitor:
    # def __init__(self, log_dir='logs', rank=0, model_type=None, model_pos_enc=None, num_iterations=None, 
    #              tokens_per_batch=None, model=None, val_tokens=0):

    def __init__(self, args, dataset_args, model_args, model, optimizer, checkpoint_step, rank=0):
        # self.num_iterations = dataset_args.iterations
        # self.num_iterations = min(dataset_args.iterations, args.num_iterations)
        self.num_iterations = args.num_iterations
        if checkpoint_step > 0:
            self.num_iterations += checkpoint_step
        self.tokens_per_batch = dataset_args.tokens_per_batch
        self.rank = rank
        self.val_tokens = dataset_args.val_tokens
        self.model = model
        self.optimizer = optimizer

        if rank==0:
            self.is_main = True
        else:
            self.is_main = False

        if self.is_main:
            # create the logging directory if it does not exist
            # log_dir = os.path.join(log_dir, model_pos_enc, model_type)
            log_dir = os.path.join(args.log_dir, args.model_size, args.position_encoding)
            os.makedirs(log_dir, exist_ok=True)
            for i in range(100):
                if os.path.exists(os.path.join(log_dir, f'version_{i:02d}')):
                    continue
                else:
                    os.makedirs(os.path.join(log_dir, f'version_{i:02d}'))
                    log_dir = os.path.join(log_dir, f'version_{i:02d}')
                    break
            else:
                raise Exception('Too many versions')
            self.log_dir = log_dir

            # self.train_log_file = f'{self.log_dir}/{model_type}_train.log'
            # with open(self.train_log_file, 'w') as f:
            #     f.write(f'step,time,loss,norm,lr,exec_time,tok/sec\n')

            # self.val_log_file = f'{self.log_dir}/{model_type}_val.log'
            # with open(self.val_log_file, 'w') as f:
            #     f.write(f'step,time,loss,exec_time,tok/sec\n')
            self.train_log_file = os.path.join(self.log_dir, f'train.log')
            self.eval_log_file = os.path.join(self.log_dir, f'eval.log')

            # with open(os.path.join(self.log_dir, 'args.log'), 'w') as f:
            #     f.write(f'arg,value\n')
            #     for args in [args, dataset_args, model_args]:
            #         for k, v in vars(args).items():
            #             f.write(f'{k},{v}\n')
            args_dict = {
                'args': vars(args),
                'dataset_args': vars(dataset_args),
                'model_args': vars(model_args)
            }
            with open(os.path.join(self.log_dir, 'args.json'), 'w') as f:
                json.dump(args_dict, f, indent=4)
            # with open(os.path.join(self.log_dir, 'model_args.json'), 'w') as f:
            #     json.dump(model_args.__dict__, f, indent=4)

            with open(self.train_log_file, 'w') as f:
                f.write(f'step,time,loss,norm,lr,exec_time,tok/sec\n')
            with open(self.eval_log_file, 'w') as f:
                f.write(f'step,time,loss,exec_time,tok/sec\n')

            
            self.log_init_time = time.time()
            self.last_log_time = self.log_init_time


    def log(self, step, loss, norm, lr):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            exec_time = time.time() - self.last_log_time
            tokens_per_second = self.tokens_per_batch / exec_time
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{norm},{lr},{exec_time},{tokens_per_second}\n'
            with open(self.train_log_file, 'a') as f:
                f.write(string)
                
            lr = lr if lr is not None else 0
            print(f"step {step+1:4d}/{self.num_iterations} | train loss {loss:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(exec_time):.2f} s | {tokens_per_second:.0f} tok/s)")
        
            if step % 256 == 0:
                # checkpoint(self.model, rank=self.rank)
                self.save_model()
                self.save_optimizer()
        
    
    def log_val(self, step, loss):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            exec_time = time.time() - self.last_log_time
            tokens_per_second = self.val_tokens / exec_time
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{exec_time},{tokens_per_second}\n'
            with open(self.eval_log_file, 'a') as f:
                f.write(string)
            # print0(f"val loss {val_loss}")
            print(f'val loss {loss:.6f} | ({exec_time:.1f}{(runtime):.1f} s | {tokens_per_second:.0f} tok/s)')
            # checkpoint(self.model, rank=self.rank)
    

    def save_model(self):
        if self.is_main:
            # state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            filename = os.path.join(self.log_dir, 'model.pt')
            torch.save(state_dict, filename)
    
    def save_optimizer(self):
        if self.is_main:
            filename = os.path.join(self.log_dir, 'optimizer.pt')
            torch.save(self.optimizer.state_dict(), filename)

    def max_memory(self):
        if self.is_main:
            mem = torch.cuda.max_memory_allocated() // 1024 / 1024
            print0(f"peak memory consumption:                       {mem:16,.2f} MiB")
            