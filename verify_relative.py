"""Verificação COMPORTAMENTAL do caminho flex_attention(score_mod, block_mask) do
relative.py. Rodar na GPU (flex_attention exige CUDA/Triton).

Não basta "rodou sem exception": aqui comparamos contra uma atenção eager de
referência, checamos fluxo de gradiente e o efeito da posição. Se algum assert
falhar, o viés relativo NÃO está sendo aplicado como deveria.

    python verify_relative.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from models.relative import relative_position_bucket, RelativeTransformer, NUM_BUCKETS, MAX_DISTANCE
from models.rotary import RotaryModelArgs

assert torch.cuda.is_available(), "rode na GPU (flex_attention precisa de CUDA/Triton)"
dev = "cuda"
torch.manual_seed(0)

B, H, L, D = 2, 4, 64, 32
q = torch.randn(B, H, L, D, device=dev)
k = torch.randn(B, H, L, D, device=dev)
v = torch.randn(B, H, L, D, device=dev)

# viés relativo (B,H,L,L) a partir de posições permutadas
positions = torch.stack([torch.randperm(L) for _ in range(B)]).to(dev)
bias_table = torch.randn(NUM_BUCKETS, H, device=dev)
rel = positions[:, None, :] - positions[:, :, None]          # (B,L,L)
bucket = relative_position_bucket(rel, NUM_BUCKETS, MAX_DISTANCE)
position_bias = F.embedding(bucket, bias_table).permute(0, 3, 1, 2).contiguous()  # (B,H,L,L)

# máscara booleana arbitrária (True = atende), sem linha toda-False
bool_mask = torch.rand(B, L, L, device=dev) > 0.3
bool_mask[:, torch.arange(L), torch.arange(L)] = True  # garante diagonal

# ---- flex_attention (o que o relative.py usa) ----
def score_mod(score, b, h, q_idx, kv_idx):
    return score + position_bias[b, h, q_idx, kv_idx]

def mask_mod(b, h, q_idx, kv_idx):
    return bool_mask[b, q_idx, kv_idx]

block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L, device=dev)
out_flex = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

# ---- referência eager: softmax(QK^T/sqrt(d) + bias) com máscara -inf ----
scores = (q @ k.transpose(-2, -1)) / (D ** 0.5) + position_bias
scores = scores.masked_fill(~bool_mask[:, None, :, :], float("-inf"))
out_eager = (F.softmax(scores, dim=-1) @ v)

max_diff = (out_flex - out_eager).abs().max().item()
print(f"[1] paridade flex vs eager: max|diff| = {max_diff:.2e}", "-> OK" if max_diff < 1e-3 else "-> FALHOU")
assert max_diff < 1e-3, "score_mod/block_mask NÃO equivalem à atenção de referência!"

# ---- [2] gradiente chega na tabela de viés relativo? ----
cfg = RotaryModelArgs(dim=128, n_layers=2, n_heads=4, ffn_dim_multiplier=2); cfg.max_seq_len = 128
model = RelativeTransformer(cfg).to(dev)
tokens = torch.randint(0, cfg.vocab_size, (B, L), device=dev)
pos = torch.stack([torch.randperm(L) for _ in range(B)]).to(dev)
amask = torch.ones(B, L, L, dtype=torch.bool, device=dev)
bm = model.build_block_mask(tokens, amask)
logits = model(tokens, positions=pos, block_mask=bm)
logits.float().sum().backward()
g = model.relative_attention_bias.weight.grad
ok_grad = g is not None and g.abs().sum().item() > 0
print(f"[2] grad na relative_attention_bias: {'OK (não-nulo)' if ok_grad else 'FALHOU (sem gradiente!)'}")
assert ok_grad, "a tabela de viés relativo não recebe gradiente — não está aprendendo!"

# ---- [3] a posição realmente importa? ----
with torch.no_grad():
    out_a = model(tokens, positions=pos, block_mask=bm)
    pos2 = torch.stack([torch.randperm(L) for _ in range(B)]).to(dev)
    out_b = model(tokens, positions=pos2, block_mask=bm)
changed = (out_a - out_b).abs().max().item()
print(f"[3] mudar positions muda a saída: max|diff| = {changed:.2e}", "-> OK" if changed > 1e-4 else "-> FALHOU")
assert changed > 1e-4, "trocar as posições não mudou nada — o viés relativo não está ativo!"

print("\nTUDO VERIFICADO: o viés relativo está correto, aprende e a posição importa.")
