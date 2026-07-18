import torch
from src.model import ModularArithmeticTransformer

model = ModularArithmeticTransformer()
x = torch.randint(0, 59, (4, 2))
batch_size = x.shape[0]

tok = model.token_embed(x)
positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
pos = model.pos_embed(positions)
h = tok + pos

attn_layer = model.transformer.layers[0].self_attn
print("in_proj_weight shape:", attn_layer.in_proj_weight.shape)
print("in_proj_bias shape:", attn_layer.in_proj_bias.shape)

d_model = model.d_model
n_heads = model.n_heads
head_dim = d_model // n_heads

qkv = torch.nn.functional.linear(h, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
q, k, v = qkv.chunk(3, dim=-1)

q = q.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)
k = k.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)

import math
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
weights = torch.nn.functional.softmax(scores, dim=-1)
print("weights shape:", weights.shape)
