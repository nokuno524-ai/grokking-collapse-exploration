import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import pandas as pd
import numpy as np
import math

# Force SDPA to use math backend so double backward is supported
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def compute_effective_rank(tensor):
    """Compute effective rank (Shannon entropy of normalized singular values)."""
    if tensor.ndim > 2:
        tensor = tensor.view(tensor.size(0), -1)
    elif tensor.ndim < 2:
        return 1.0 # vectors have rank 1

    s = torch.linalg.svdvals(tensor)
    s = s / (s.sum() + 1e-10)
    entropy = -(s * torch.log(s + 1e-10)).sum()
    return torch.exp(entropy).item()

def approximate_hessian_eigenvalues(model, train_loader, device, num_iters=10):
    """Power iteration to find the top eigenvalue of the Hessian."""
    # We must be in train mode with specific configuration to avoid double backward errors
    model.train()

    # We only care about the parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]

    # Initialize random vector v
    v = [torch.randn(p.size(), device=device) for p in params]
    # Normalize v
    norm = sum(torch.sum(x**2) for x in v)**0.5
    v = [x / norm for x in v]

    eigenvalue = 0.0

    for _ in range(num_iters):
        # Compute Hessian-vector product
        Hv = [torch.zeros_like(p) for p in params]
        total_samples = 0

        # Take a subset of data for speed
        subset_batches = 3

        iterator = iter(train_loader)
        for _ in range(subset_batches):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass with no_grad contexts where needed, but we need grad here
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)

            # Compute first derivative
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            # Compute dot product v^T g
            v_dot_g = sum(torch.sum(g * x) for g, x in zip(grads, v))

            # Compute second derivative (H*v)
            Hv_batch = torch.autograd.grad(v_dot_g, params, retain_graph=False)

            for i, hv in enumerate(Hv_batch):
                Hv[i] += hv * inputs.size(0)

            total_samples += inputs.size(0)

        Hv = [x / max(total_samples, 1) for x in Hv]

        # Rayleigh quotient
        eigenvalue = sum(torch.sum(hv * x) for hv, x in zip(Hv, v)).item()

        # Normalize new v
        norm = sum(torch.sum(x**2) for x in Hv)**0.5
        v = [x / max(norm, 1e-10) for x in Hv]

    return eigenvalue

def compute_neuron_importance(model, train_loader, device):
    """Compute neuron importance scores via magnitude * activation for the FFN hidden layer."""
    model.eval()

    # We will hook into the GELU activation to get the FFN hidden states
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hook_handle = model.transformer.layers[0].linear1.register_forward_hook(hook_fn)

    params = [model.transformer.layers[0].linear1.weight]

    # Take a single batch
    iterator = iter(train_loader)
    try:
        inputs, targets = next(iterator)
    except StopIteration:
        return 0.0

    inputs, targets = inputs.to(device), targets.to(device)

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)

    # Compute gradients
    loss.backward()

    # Extract activation
    act = activations[0]

    # Grad of linear1.weight is (d_ff, d_model)
    grad = model.transformer.layers[0].linear1.weight.grad
    if grad is None:
        hook_handle.remove()
        return 0.0

    mean_abs_act = act.abs().mean(dim=(0, 1)) # (d_ff,)
    grad_norm = grad.norm(dim=1) # (d_ff,)
    importance = mean_abs_act * grad_norm # (d_ff,)

    hook_handle.remove()

    imp_squared_sum = (importance ** 2).sum()
    sum_imp_squared = importance.sum() ** 2
    if imp_squared_sum > 0:
        participation_ratio = (sum_imp_squared / imp_squared_sum).item()
    else:
        participation_ratio = importance.size(0)

    return participation_ratio

def compute_weight_forensics(condition_dir: Path, output_file: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(condition_dir / "results.json") as f:
        results = json.load(f)
    config = results['config']

    # Load dataset for Hessian and activations
    data_config = DatasetConfig(
        prime=config.get('prime', 59),
        train_fraction=config.get('train_fraction', 0.3),
        collapse_level=config.get('collapse_level', 0.0),
        collapse_severity=config.get('collapse_severity', 0.5),
        noise_fraction=config.get('noise_fraction', 0.0),
        seed=config.get('seed', 42),
    )
    train_in, train_tgt, _, _ = generate_modular_arithmetic(data_config)
    train_loader = DataLoader(TensorDataset(train_in, train_tgt), batch_size=256, shuffle=True)

    ckpts = list(condition_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {condition_dir}")
        return

    ckpts.sort(key=lambda x: int(x.stem.split('_')[1]))

    records = []
    first_ckpt_state = None

    for i, ckpt_path in enumerate(ckpts):
        step = int(ckpt_path.stem.split('_')[1])
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = ModularArithmeticTransformer(
            prime=config.get('prime', 59),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 512),
            n_layers=config.get('n_layers', 1),
        )
        state_dict_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
        state = checkpoint[state_dict_key]
        model.load_state_dict(state)
        model.to(device)

        if first_ckpt_state is None:
            first_ckpt_state = state

        record = {'step': step}

        total_dist_sq = 0.0
        for name, tensor in state.items():
            if 'weight' in name or 'bias' in name:
                diff = tensor - first_ckpt_state[name]
                total_dist_sq += torch.sum(diff ** 2).item()
        record['dist_from_init'] = math.sqrt(total_dist_sq) if 'math' in sys.modules else total_dist_sq**0.5

        record['embed_rank'] = compute_effective_rank(state['token_embed.weight'])

        attn_out_key = 'transformer.layers.0.self_attn.out_proj.weight'
        if attn_out_key in state:
            record['attn_out_rank'] = compute_effective_rank(state[attn_out_key])

        ffn_in_key = 'transformer.layers.0.linear1.weight'
        if ffn_in_key in state:
            record['ffn_in_rank'] = compute_effective_rank(state[ffn_in_key])

        out_head_key = 'output_head.weight'
        if out_head_key in state:
            record['out_head_rank'] = compute_effective_rank(state[out_head_key])

        try:
            record['top_hessian_eig'] = approximate_hessian_eigenvalues(model, train_loader, device)
        except Exception as e:
            print(f"Failed to compute Hessian at step {step}: {e}")
            record['top_hessian_eig'] = float('nan')

        record['neuron_participation_ratio'] = compute_neuron_importance(model, train_loader, device)

        records.append(record)
        print(f"[{condition_dir.name}] Step {step} processed.")

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Saved forensics to {output_file}")

def main():
    base_dir = Path("results")
    out_dir = Path("analysis/weight_forensics")
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["pure", "medium_collapse", "severe_collapse"]
    for c in conditions:
        cond_dir = base_dir / c
        if cond_dir.exists():
            print(f"Processing condition {c}...")
            out_file = out_dir / f"{c}_forensics.csv"
            compute_weight_forensics(cond_dir, out_file)
        else:
            print(f"Condition dir {cond_dir} not found.")

if __name__ == "__main__":
    main()
