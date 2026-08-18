import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer

model = ModularArithmeticTransformer()
model.eval()
x = torch.randint(0, 59, (4, 2))

# forward to get attention weights. The memory note says:
# To extract multi-head attention weights from ModularArithmeticTransformer, the Q, K, V projections must be manually reconstructed from transformer.layers.0.self_attn.in_proj_weight, as the default PyTorch TransformerEncoderLayer hardcodes need_weights=False.

tok = model.token_embed(x)
pos = model.pos_embed(torch.arange(2).unsqueeze(0).expand(4, -1))
h = tok + pos
layer = model.transformer.layers[0]

# manual forward
in_proj_weight = layer.self_attn.in_proj_weight
in_proj_bias = layer.self_attn.in_proj_bias

# shape h: (batch, seq, d_model) -> (batch*seq, d_model) for linear
qkv = torch.nn.functional.linear(h, in_proj_weight, in_proj_bias)
q, k, v = qkv.chunk(3, dim=-1)

# reshape to (batch, n_heads, seq, d_head)
batch_size, seq_len, _ = h.shape
n_heads = model.n_heads
d_head = model.d_model // n_heads
q = q.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
k = k.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
v = v.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

# scores
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
attn_weights = torch.softmax(attn_scores, dim=-1)

print(attn_weights.shape)
