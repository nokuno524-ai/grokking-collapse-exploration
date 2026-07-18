import torch
import torch.nn.functional as F
from src.model import ModularArithmeticTransformer

def get_attention_weights(model, x):
    # This matches the model forward pass and intercepts attention weights
    batch_size = x.shape[0]

    # Token embeddings
    tok = model.token_embed(x)  # (batch, 2, d_model)

    # Positional embeddings
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)  # (batch, 2, d_model)

    # Combine
    h = tok + pos  # (batch, 2, d_model)

    # We need to manually do the attention to get the weights
    attn_layer = model.transformer.layers[0].self_attn
    qkv = F.linear(h, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    n_heads = model.n_heads
    d_model = model.d_model
    head_dim = d_model // n_heads

    q = q.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)

    import math
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    weights = F.softmax(scores, dim=-1)

    return weights

model = ModularArithmeticTransformer()
x = torch.randint(0, 59, (4, 2))
weights = get_attention_weights(model, x)
print(weights.shape)
