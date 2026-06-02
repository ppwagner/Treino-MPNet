"""Variante de posição ABSOLUTA do modelo (espelha models/rotary.py).

Idêntico ao RotaryTransformer EXCETO pela codificação posicional:
  - sem RoPE (sem precompute_freqs_cis / apply_rotary_emb);
  - embeddings de posição APRENDIDOS (nn.Embedding) somados ao token embedding,
    indexados pelos MESMOS `positions` permutados/compensados que o dataset emite.

Reaproveita RMSNorm, FeedForward, repeat_kv e RotaryModelArgs do rotary.py para
garantir que TODA a arquitetura (dim, camadas, FFN, norm, estrutura de atenção)
seja a mesma — só a posição muda. Assim a comparação rope-vs-absolute isola o RoPE.

Interface idêntica à do RotaryTransformer: build_block_mask(...) + forward(tokens,
positions=, block_mask=) -> logits.
"""
from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from models.rotary import RMSNorm, FeedForward, RotaryModelArgs, repeat_kv


class Attention(nn.Module):
    """Igual ao Attention do rotary.py, mas SEM aplicar RoPE em Q/K."""

    def __init__(self, args: RotaryModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # >>> única diferença vs rotary.py: NÃO aplicamos apply_rotary_emb aqui <<<
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = flex_attention(
            queries,
            keys,
            values,
            block_mask=mask,
            kernel_options={
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
            },
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: RotaryModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class AbsoluteTransformer(nn.Module):
    def __init__(self, params: RotaryModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # posição absoluta aprendida; indexada pelos valores de `positions` (< seg_len),
        # que cabem em max_seq_len (definido em train.py como int(seq_len * 1.5)).
        self.pos_embeddings = nn.Embedding(params.max_seq_len, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def build_block_mask(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_codes: Optional[torch.Tensor] = None,
    ):
        """Idêntico ao RotaryTransformer.build_block_mask: constrói o BlockMask em
        modo eager (fora do torch.compile) a partir da máscara two-stream."""
        bsz, seqlen = tokens.shape

        seq_codes = (
            seq_codes
            if seq_codes is not None
            else torch.zeros_like(tokens, device=tokens.device)
        )

        def mask_mod(b, h, q_idx, kv_idx):
            return attention_mask[b, q_idx, kv_idx]

        block_mask = create_block_mask(
            mask_mod,
            B=bsz,
            H=None,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=tokens.device,
            BLOCK_SIZE=128,
        )
        return block_mask

    def forward(
        self,
        tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        block_mask=None,
        **kwargs,
    ):
        bsz, seqlen = tokens.shape
        if positions is None:
            positions = torch.arange(seqlen, device=tokens.device).unsqueeze(0).expand(bsz, seqlen)

        # >>> diferença vs rotary.py: posição entra aqui (soma), não via RoPE na atenção <<<
        h = self.tok_embeddings(tokens) + self.pos_embeddings(positions)

        for layer in self.layers:
            h = layer(h, block_mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
