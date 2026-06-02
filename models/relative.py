"""Variante de posição RELATIVA (estilo T5/MPNet) — espelha models/rotary.py.

Opção (i) da ablação: posição entra SÓ como viés relativo aditivo nos scores de
atenção. SEM RoPE e SEM embedding absoluto — assim a comparação isola "RoPE vs
relativo" (cada variante = um único mecanismo posicional).

Mecânica do viés relativo (idêntica à do HF MPNet / T5):
  - para cada par (query i, key j): d = pos[j] - pos[i];
  - d é bucketizado (relative_position_bucket): exato p/ distâncias pequenas,
    logarítmico p/ grandes; metade dos baldes p/ d<0, metade p/ d>0;
  - tabela aprendida relative_attention_bias: (num_buckets -> n_heads) escalar;
  - o viés (B, H, L, L) é somado aos scores via score_mod do flex_attention,
    combinado com o block_mask two-stream que o dataset já produz.
  - calculado UMA vez no forward e compartilhado por todas as camadas (como no T5).

Reaproveita RMSNorm, FeedForward, repeat_kv e RotaryModelArgs do rotary.py para
manter a arquitetura byte-idêntica às outras variantes. Interface igual:
build_block_mask(...) + forward(tokens, positions=, block_mask=) -> logits.
"""
import math
from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from models.rotary import RMSNorm, FeedForward, RotaryModelArgs, repeat_kv

# Hiperparâmetros do viés relativo (defaults do HF MPNet / T5).
NUM_BUCKETS = 32
MAX_DISTANCE = 128


def relative_position_bucket(relative_position, num_buckets=NUM_BUCKETS, max_distance=MAX_DISTANCE):
    """Bucketização bidirecional do T5/MPNet (réplica do HF MPNet.relative_position_bucket)."""
    ret = 0
    n = -relative_position
    num_buckets //= 2
    ret += (n < 0).to(torch.long) * num_buckets
    n = torch.abs(n)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


class Attention(nn.Module):
    """Igual ao Attention do rotary.py, mas SEM RoPE e COM viés relativo (score_mod)."""

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], position_bias: torch.Tensor):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # >>> diferença vs rotary.py: sem apply_rotary_emb; posição vira viés nos scores <<<
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + position_bias[b, h, q_idx, kv_idx]

        output = flex_attention(
            queries,
            keys,
            values,
            score_mod=score_mod,
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], position_bias: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), mask, position_bias)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class RelativeTransformer(nn.Module):
    def __init__(self, params: RotaryModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_heads = params.n_heads

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # tabela de viés relativo: 1 escalar por (balde, cabeça); compartilhada entre camadas.
        self.relative_attention_bias = nn.Embedding(NUM_BUCKETS, params.n_heads)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def compute_position_bias(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, L) -> viés (B, H, L, L) somado aos scores de atenção."""
        context_position = positions[:, :, None]   # (B, L, 1)
        memory_position = positions[:, None, :]     # (B, 1, L)
        relative_position = memory_position - context_position  # (B, L, L)
        rp_bucket = relative_position_bucket(relative_position)  # (B, L, L)
        values = self.relative_attention_bias(rp_bucket)         # (B, L, L, H)
        return values.permute(0, 3, 1, 2).contiguous()           # (B, H, L, L)

    def build_block_mask(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_codes: Optional[torch.Tensor] = None,
    ):
        """Idêntico às outras variantes: BlockMask two-stream em modo eager."""
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

        # sem posição no embedding (T5-style); ela entra só pelo viés relativo na atenção
        h = self.tok_embeddings(tokens)
        position_bias = self.compute_position_bias(positions)  # uma vez, compartilhado

        for layer in self.layers:
            h = layer(h, block_mask, position_bias)
        h = self.norm(h)
        output = self.output(h).float()
        return output
