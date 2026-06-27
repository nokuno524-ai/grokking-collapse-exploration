import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import math
import pandas as pd
from typing import Dict, List, Optional
import numpy as np

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def replace_attention_with_gated(model):
    original_attn = model.transformer.layers[0].self_attn

    # We will subclass MultiheadAttention to gate the heads *before* the output projection.
    class GatedMultiheadAttention(type(original_attn)):
        def __init__(self, original):
            super().__init__(
                embed_dim=original.embed_dim,
                num_heads=original.num_heads,
                dropout=original.dropout,
                bias=original.in_proj_bias is not None,
                add_bias_kv=original.bias_k is not None,
                add_zero_attn=original.add_zero_attn,
                kdim=original.kdim,
                vdim=original.vdim,
                batch_first=original.batch_first
            )
            self.load_state_dict(original.state_dict())
            self.gates = nn.Parameter(torch.ones(self.num_heads))

        def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
            # We rewrite the final projection part of the attention

            if self.batch_first:
                batch_size = query.size(0)
                seq_len = query.size(1)
            else:
                batch_size = query.size(1)
                seq_len = query.size(0)
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)

            head_dim = self.embed_dim // self.num_heads

            # Using in_proj_weight / in_proj_bias
            if self.in_proj_weight is not None:
                qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                q = F.linear(query, self.q_proj_weight)
                k = F.linear(key, self.k_proj_weight)
                v = F.linear(value, self.v_proj_weight)

            q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)

            context = torch.matmul(attn, v) # (batch, num_heads, seq_len, head_dim)

            # Apply gating HERE, before reshaping and out_proj
            gates_view = self.gates.view(1, self.num_heads, 1, 1)
            context = context * gates_view

            context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

            output = self.out_proj(context)

            if not self.batch_first:
                output = output.transpose(0, 1)

            return output, attn

    gated_attn = GatedMultiheadAttention(original_attn)
    model.transformer.layers[0].self_attn = gated_attn

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze gates
    model.transformer.layers[0].self_attn.gates.requires_grad = True
    return model

def train_gates(model, train_loader, device, num_steps=500, lambda_l1=0.01, lr=0.1):
    model.to(device)
    model.eval() # keep other parts in eval mode (e.g. dropout, LN)
    gates_param = model.transformer.layers[0].self_attn.gates
    optimizer = torch.optim.Adam([gates_param], lr=lr)

    with torch.no_grad():
        gates_param.add_(torch.randn_like(gates_param) * 0.01)

    iterator = iter(train_loader)
    for step in range(num_steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        ce_loss = F.cross_entropy(logits, targets)

        l1_loss = lambda_l1 * torch.sum(torch.abs(gates_param))

        loss = ce_loss + l1_loss
        loss.backward()

        optimizer.step()

    return gates_param.detach().cpu().numpy()

def analyze_gating_trajectory(condition_dir: Path, output_file: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(condition_dir / "results.json") as f:
        results = json.load(f)
    config = results['config']

    # Load dataset
    data_config = DatasetConfig(
        prime=config.get('prime', 59),
        train_fraction=config.get('train_fraction', 0.3),
        collapse_level=config.get('collapse_level', 0.0),
        collapse_severity=config.get('collapse_severity', 0.5),
        noise_fraction=config.get('noise_fraction', 0.0),
        seed=config.get('seed', 42),
    )
    train_in, train_tgt, _, _ = generate_modular_arithmetic(data_config)
    train_loader = DataLoader(TensorDataset(train_in, train_tgt), batch_size=512, shuffle=True)

    ckpts = list(condition_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {condition_dir}")
        return

    ckpts.sort(key=lambda x: int(x.stem.split('_')[1]))

    records = []

    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.split('_')[1])
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = ModularArithmeticTransformer(
            prime=config.get('prime', 59),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 512),
            n_layers=config.get('n_layers', 1),
        )

        # Handle "model_state_dict" vs "model_state"
        state_dict_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
        model.load_state_dict(checkpoint[state_dict_key])

        model = replace_attention_with_gated(model)

        # Initialize gates to 1
        model.transformer.layers[0].self_attn.gates.data.fill_(1.0)

        gates_val = train_gates(model, train_loader, device, num_steps=200, lambda_l1=0.005, lr=0.05)

        record = {'step': step}
        for h, g in enumerate(gates_val):
            record[f'head_{h}'] = g
        records.append(record)
        print(f"[{condition_dir.name}] Step {step}: gates = {gates_val}")

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

def main():
    base_dir = Path("results")
    out_dir = Path("analysis/causal_head_gating")
    out_dir.mkdir(parents=True, exist_ok=True)

    # We will run this on pure, medium_collapse, and severe_collapse
    conditions = ["pure", "medium_collapse", "severe_collapse"]
    for c in conditions:
        cond_dir = base_dir / c
        if cond_dir.exists():
            print(f"Processing condition {c}...")
            out_file = out_dir / f"{c}_gates.csv"
            analyze_gating_trajectory(cond_dir, out_file)
        else:
            print(f"Condition dir {cond_dir} not found.")

if __name__ == "__main__":
    main()
