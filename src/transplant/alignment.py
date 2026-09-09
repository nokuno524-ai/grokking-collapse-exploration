import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
from typing import Tuple, Dict, Any, List

def compute_activation_correlation(act_a: torch.Tensor, act_b: torch.Tensor) -> np.ndarray:
    """
    Compute Pearson correlation between columns (neurons/heads) of act_a and act_b.
    act_a, act_b shape: (batch_size * seq_len, num_features)
    Returns: (num_features, num_features) correlation matrix.
    """
    act_a = act_a - act_a.mean(dim=0, keepdim=True)
    act_b = act_b - act_b.mean(dim=0, keepdim=True)

    norm_a = act_a.norm(dim=0, keepdim=True) + 1e-8
    norm_b = act_b.norm(dim=0, keepdim=True) + 1e-8

    act_a_norm = act_a / norm_a
    act_b_norm = act_b / norm_b

    corr = torch.mm(act_a_norm.t(), act_b_norm).cpu().detach().numpy()
    return corr

def align_mlp_layer(w1_a, w1_b, w2_a, w2_b, b1_a, b1_b, act_a, act_b) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Align MLP neurons based on activation correlation."""
    corr = compute_activation_correlation(act_a, act_b)
    cost_matrix = -corr
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    permuted_w1_b = w1_b[col_ind, :]
    permuted_b1_b = b1_b[col_ind] if b1_b is not None else None
    permuted_w2_b = w2_b[:, col_ind]
    return permuted_w1_b, permuted_w2_b, permuted_b1_b, float(corr[row_ind, col_ind].mean())

def align_attention_heads(in_proj_a, in_proj_b, out_proj_a, out_proj_b, in_bias_b, act_a, act_b, n_heads: int, d_model: int):
    """Align attention heads based on head output activations."""
    head_dim = d_model // n_heads
    N = act_a.shape[0]
    act_a = act_a.view(N, n_heads, head_dim)
    act_b = act_b.view(N, n_heads, head_dim)

    # We want to match heads, so treat each head's activation as a flat vector of N * head_dim
    act_a_flat = act_a.permute(1, 0, 2).reshape(n_heads, N * head_dim).t()
    act_b_flat = act_b.permute(1, 0, 2).reshape(n_heads, N * head_dim).t()

    corr = compute_activation_correlation(act_a_flat, act_b_flat)
    cost_matrix = -corr
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    def permute_in_proj(in_proj, perm):
        q, k, v = in_proj.chunk(3, dim=0)
        def permute_qkv(w, perm):
            w = w.view(n_heads, head_dim, -1)
            w = w[perm, :, :]
            return w.view(d_model, -1)
        q_p = permute_qkv(q, perm)
        k_p = permute_qkv(k, perm)
        v_p = permute_qkv(v, perm)
        return torch.cat([q_p, k_p, v_p], dim=0)

    def permute_out_proj(out_proj, perm):
        w = out_proj.view(d_model, n_heads, head_dim)
        w = w[:, perm, :]
        return w.view(d_model, d_model)

    perm_in_proj_b = permute_in_proj(in_proj_b, col_ind)
    perm_out_proj_b = permute_out_proj(out_proj_b, col_ind)

    perm_in_bias_b = None
    if in_bias_b is not None:
        perm_in_bias_b = permute_in_proj(in_bias_b.unsqueeze(-1), col_ind).squeeze(-1)

    return perm_in_proj_b, perm_out_proj_b, perm_in_bias_b, float(corr[row_ind, col_ind].mean())

def get_attention_head_outputs(layer, x_attn):
    """Manually compute pre-out_proj attention head activations."""
    B, T, C = x_attn.shape
    H = layer.self_attn.num_heads
    D = C // H

    qkv = torch.nn.functional.linear(x_attn, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    attn = torch.softmax(scores, dim=-1)

    head_outs = torch.matmul(attn, v) # (B, H, T, D)
    head_outs_concat = head_outs.transpose(1, 2).reshape(B * T, H * D)
    return head_outs_concat

def collect_activations(model: nn.Module, dataloader) -> Dict[str, torch.Tensor]:
    """Collect intermediate activations across the dataset."""
    acts = {}
    hooks = []

    attn_inputs = {}

    def get_hook(name):
        acts[name] = []
        def hook(model, input, output):
            val = output[0] if isinstance(output, tuple) else output
            acts[name].append(val.detach().cpu())
        return hook

    def get_attn_input_hook(name):
        attn_inputs[name] = []
        def hook(module, input, output):
            attn_inputs[name].append(input[0].detach().cpu())
        return hook

    for i in range(model.n_layers if hasattr(model, 'n_layers') else 1):
        layer = model.transformer.layers[i]
        hooks.append(layer.linear1.register_forward_hook(get_hook(f"layer_{i}_mlp")))
        hooks.append(layer.self_attn.register_forward_hook(get_attn_input_hook(f"layer_{i}_attn_in")))

    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i > 20:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    final_acts = {}
    for k, v in acts.items():
        cat_v = torch.cat(v, dim=0)
        final_acts[k] = cat_v.view(-1, cat_v.shape[-1])

    for i in range(model.n_layers if hasattr(model, 'n_layers') else 1):
        attn_in = torch.cat(attn_inputs[f"layer_{i}_attn_in"], dim=0)
        layer = model.transformer.layers[i]
        head_outs = get_attention_head_outputs(layer, attn_in)
        final_acts[f"layer_{i}_attn"] = head_outs

    return final_acts

def align_models(model_a: nn.Module, model_b: nn.Module, dataloader=None) -> Tuple[nn.Module, float, float]:
    """
    Align model_b to model_a using activation matching.
    Returns:
        aligned_model_b (nn.Module)
        mean_corr_before (float)
        mean_corr_after (float)
    """
    if dataloader is None:
        device = next(model_a.parameters()).device
        x = torch.randint(0, model_a.prime, (1024, 2))
        y = torch.zeros(1024, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)

    aligned_b = copy.deepcopy(model_b)

    model_a.eval()
    model_b.eval()
    acts_a = collect_activations(model_a, dataloader)
    acts_b = collect_activations(model_b, dataloader)

    sd_a = model_a.state_dict()
    sd_b = aligned_b.state_dict()

    d_model = model_a.d_model
    n_heads = model_a.n_heads

    sims_before = []
    sims_after = []

    for i in range(model_a.n_layers if hasattr(model_a, 'n_layers') else 1):
        # MLP
        w1_a = sd_a[f'transformer.layers.{i}.linear1.weight']
        w1_b = sd_b[f'transformer.layers.{i}.linear1.weight']
        w2_a = sd_a[f'transformer.layers.{i}.linear2.weight']
        w2_b = sd_b[f'transformer.layers.{i}.linear2.weight']
        b1_a = sd_a.get(f'transformer.layers.{i}.linear1.bias', None)
        b1_b = sd_b.get(f'transformer.layers.{i}.linear1.bias', None)

        act_a_mlp = acts_a[f'layer_{i}_mlp']
        act_b_mlp = acts_b[f'layer_{i}_mlp']

        corr_before = compute_activation_correlation(act_a_mlp, act_b_mlp)
        sim_before_mlp = float(corr_before.diagonal().mean())
        sims_before.append(sim_before_mlp)

        permuted_w1, permuted_w2, permuted_b1, sim_after_mlp = align_mlp_layer(
            w1_a, w1_b, w2_a, w2_b, b1_a, b1_b, act_a_mlp, act_b_mlp)
        sims_after.append(sim_after_mlp)

        sd_b[f'transformer.layers.{i}.linear1.weight'] = permuted_w1
        sd_b[f'transformer.layers.{i}.linear2.weight'] = permuted_w2
        if permuted_b1 is not None:
            sd_b[f'transformer.layers.{i}.linear1.bias'] = permuted_b1

        # Attention
        in_a = sd_a[f'transformer.layers.{i}.self_attn.in_proj_weight']
        in_b = sd_b[f'transformer.layers.{i}.self_attn.in_proj_weight']
        out_a = sd_a[f'transformer.layers.{i}.self_attn.out_proj.weight']
        out_b = sd_b[f'transformer.layers.{i}.self_attn.out_proj.weight']
        in_bias_b = sd_b.get(f'transformer.layers.{i}.self_attn.in_proj_bias', None)

        act_a_attn = acts_a[f'layer_{i}_attn']
        act_b_attn = acts_b[f'layer_{i}_attn']

        head_dim = d_model // n_heads

        N = act_a_attn.shape[0]
        aa_flat = act_a_attn.view(N, n_heads, head_dim).permute(1, 0, 2).reshape(n_heads, -1).t()
        ab_flat = act_b_attn.view(N, n_heads, head_dim).permute(1, 0, 2).reshape(n_heads, -1).t()
        c_before_attn = compute_activation_correlation(aa_flat, ab_flat)
        sim_before_attn = float(c_before_attn.diagonal().mean())
        sims_before.append(sim_before_attn)

        perm_in, perm_out, perm_in_bias_b, sim_after_attn = align_attention_heads(
            in_a, in_b, out_a, out_b, in_bias_b, act_a_attn, act_b_attn, n_heads, d_model)
        sims_after.append(sim_after_attn)

        sd_b[f'transformer.layers.{i}.self_attn.in_proj_weight'] = perm_in
        sd_b[f'transformer.layers.{i}.self_attn.out_proj.weight'] = perm_out
        if perm_in_bias_b is not None:
            sd_b[f'transformer.layers.{i}.self_attn.in_proj_bias'] = perm_in_bias_b

    aligned_b.load_state_dict(sd_b, strict=True)
    return aligned_b, float(np.mean(sims_before)), float(np.mean(sims_after))
