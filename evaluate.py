import torch
import argparse
from eval_utils import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
parser.add_argument('--seq_batch_size', type=int, default=32, help='Sequence batch size for KV cache evaluation')
parser.add_argument('--linspace_start', type=int, default=0, help='Start of the linspace for passkey sequence lengths')
parser.add_argument('--linspace_end', type=int, default=32_768, help='End of the linspace for passkey sequence lengths')
parser.add_argument('--linspace_steps', type=int, default=10, help='Number of steps in the linspace for passkey sequence lengths')
parser.add_argument('--passkey_sample_size', type=int, default=20, help='Number of samples for passkey evaluation')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to look for model checkpoints')
parser.add_argument('--perplexity', action=argparse.BooleanOptionalAction, default=False, help='Whether to run perplexity evaluation')
parser.add_argument('--perplexity_seq_len', type=int, default=32_768, help='Sequence length for perplexity evaluation')
parser.add_argument('--perplexity_wiki_articles', type=int, default=128, help='Number of Wikipedia articles for perplexity evaluation')
args = parser.parse_args()

if args.perplexity:
    perplexity_dataset_dirs = ['wikipedia'],

passkey_seq_lens = torch.linspace(args.linspace_start, args.linspace_end, args.linspace_steps).int().tolist()
evaluator = Evaluator(device=args.device,
                      seq_batch_size=args.seq_batch_size,
                      passkey_seq_lens=passkey_seq_lens,
                      passkey_sample_size=args.passkey_sample_size,
                      perplexity_seq_len=args.perplexity_seq_len,
                      perplexity_wiki_articles=args.perplexity_wiki_articles,
                      perplexity_dataset_dirs=['wikipedia'] if args.perplexity else [],
)
# bam_ssmax_1024      = 'logs/l12/bam_ssmax/version_08/'
evaluator.evaluate(args.log_dir)
