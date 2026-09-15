import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy
import re

def transplant_attention_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int
) -> Dict[str, torch.Tensor]:
    """Transplants a specific attention head from donor to base."""
    out = {k: v.clone() for k, v in base_sd.items()}
    head_dim = d_model // n_heads
    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    # MultiheadAttention fused in_proj_weight is shape (3*d_model, d_model)
    in_proj_w_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_b_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"

    if in_proj_w_key in base_sd and in_proj_w_key in donor_sd:
        for offset in [0, d_model, 2 * d_model]: # Q, K, V
            # Weight
            out[in_proj_w_key][offset + start_idx : offset + end_idx, :] = \
                donor_sd[in_proj_w_key][offset + start_idx : offset + end_idx, :].clone()

            # Bias
            if in_proj_b_key in base_sd and in_proj_b_key in donor_sd:
                out[in_proj_b_key][offset + start_idx : offset + end_idx] = \
                    donor_sd[in_proj_b_key][offset + start_idx : offset + end_idx].clone()

    # out_proj is (d_model, d_model), mapping concatenated head outputs back
    out_proj_w_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"

    if out_proj_w_key in base_sd and out_proj_w_key in donor_sd:
        # For out_proj weight, columns correspond to the output of different heads
        out[out_proj_w_key][:, start_idx : end_idx] = \
            donor_sd[out_proj_w_key][:, start_idx : end_idx].clone()

    return out
