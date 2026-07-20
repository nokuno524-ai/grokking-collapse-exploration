import os
import json
import torch
import math
import numpy as np
from glob import glob
from pathlib import Path

try:
    from src.model import ModularArithmeticTransformer
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import ModularArithmeticTransformer


def evaluate_model(model, data):
    """Evaluate model accuracy on given data."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=-1)
        targets = (data[:, 0] + data[:, 1]) % model.prime
        return (preds == targets).float().mean().item()


def get_activations(model, x):
    """
    Get activations from different layers of the model.
    """
    acts = {}

    # Token embeddings
    tok = model.token_embed(x)
    acts['tok'] = tok

    # Positional embeddings
    batch_size, seq_len = x.shape
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    acts['pos'] = pos

    # Combined
    h = tok + pos
    acts['resid_pre'] = h

    # Inside transformer layer
    layer = model.transformer.layers[0]

    h_norm = layer.norm1(h)

    # Self attention
    attn_out, _ = layer.self_attn(h_norm, h_norm, h_norm, need_weights=False)
    acts['attn_out'] = attn_out

    h = h + layer.dropout1(attn_out)
    acts['resid_mid'] = h

    h_norm2 = layer.norm2(h)

    # Feed forward
    ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2))))
    acts['ff_out'] = ff_out

    h = h + layer.dropout2(ff_out)
    acts['resid_post'] = h

    # Output head
    h_pool = model.ln(h).mean(dim=1)
    logits = model.output_head(h_pool)
    acts['logits'] = logits

    return acts


def patch_activations(model, clean_x, corrupt_x, patch_point, clean_acts):
    """
    Run model on corrupt_x, but replace activations at patch_point with those from clean_acts.
    """
    model.eval()
    batch_size, seq_len = corrupt_x.shape

    with torch.no_grad():
        tok = model.token_embed(corrupt_x)
        positions = torch.arange(seq_len, device=corrupt_x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)

        h = tok + pos

        if patch_point == 'resid_pre':
            h = clean_acts['resid_pre']

        layer = model.transformer.layers[0]

        h_norm = layer.norm1(h)
        attn_out, _ = layer.self_attn(h_norm, h_norm, h_norm, need_weights=False)

        if patch_point == 'attn_out':
            attn_out = clean_acts['attn_out']

        h = h + layer.dropout1(attn_out)

        if patch_point == 'resid_mid':
            h = clean_acts['resid_mid']

        h_norm2 = layer.norm2(h)
        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2))))

        if patch_point == 'ff_out':
            ff_out = clean_acts['ff_out']

        h = h + layer.dropout2(ff_out)

        if patch_point == 'resid_post':
            h = clean_acts['resid_post']

        h_pool = model.ln(h).mean(dim=1)
        logits = model.output_head(h_pool)

        preds = logits.argmax(dim=-1)
        targets = (clean_x[:, 0] + clean_x[:, 1]) % model.prime
        acc = (preds == targets).float().mean().item()

    return acc


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = ckpt.get('config', {'prime': 59, 'd_model': 128, 'n_heads': 4, 'd_ff': 512})
    model = ModularArithmeticTransformer(
        prime=config.get('prime', 59),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 512)
    )
    model.load_state_dict(ckpt['model_state'])
    return model


def analyze_circuit_completeness(grokked_ckpt_path, ungrokked_ckpt_path):
    """
    Measure how much of the task is solved by identified circuits
    by patching from grokked to ungrokked model.
    """
    model_g = load_model(grokked_ckpt_path)
    model_u = load_model(ungrokked_ckpt_path)

    prime = model_g.prime

    # Generate clean dataset
    a = torch.arange(prime)
    b = torch.arange(prime)
    A, B = torch.meshgrid(a, b, indexing='ij')
    clean_x = torch.stack([A.flatten(), B.flatten()], dim=1)

    # Generate corrupt dataset (random labels)
    corrupt_x = torch.randint(0, prime, (len(clean_x), 2))

    # Get clean activations
    model_g.eval()
    with torch.no_grad():
        clean_acts = get_activations(model_g, clean_x)

    # Patch and evaluate
    patch_points = ['resid_pre', 'attn_out', 'resid_mid', 'ff_out', 'resid_post']
    results = {}

    for point in patch_points:
        acc = patch_activations(model_u, clean_x, corrupt_x, point, clean_acts)
        results[point] = acc

    return results

def track_circuit_formation(run_dir):
    """
    Track emergence of modular addition circuits by evaluating patching accuracy
    from the final grokked model to intermediate checkpoints.
    """
    ckpts = sorted(glob(os.path.join(run_dir, 'checkpoint_*.pt')),
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not ckpts:
        return {}, []

    final_ckpt = ckpts[-1]
    model_final = load_model(final_ckpt)

    prime = model_final.prime
    a = torch.arange(prime)
    b = torch.arange(prime)
    A, B = torch.meshgrid(a, b, indexing='ij')
    clean_x = torch.stack([A.flatten(), B.flatten()], dim=1)
    corrupt_x = torch.randint(0, prime, (len(clean_x), 2))

    model_final.eval()
    with torch.no_grad():
        clean_acts = get_activations(model_final, clean_x)

    trajectories = {
        'attn_out': [],
        'ff_out': []
    }
    steps = []

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        step = ckpt.get('step', int(ckpt_path.split('_')[-1].split('.')[0]))

        model_t = load_model(ckpt_path)

        acc_attn = patch_activations(model_t, clean_x, corrupt_x, 'attn_out', clean_acts)
        acc_ff = patch_activations(model_t, clean_x, corrupt_x, 'ff_out', clean_acts)

        trajectories['attn_out'].append(acc_attn)
        trajectories['ff_out'].append(acc_ff)
        steps.append(step)

    return trajectories, steps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pure_dir', type=str, default="results/pure")
    parser.add_argument('--collapse_dir', type=str, default="results/severe_collapse")
    args = parser.parse_args()

    for condition_name, condition_dir in [("Pure", args.pure_dir), ("Collapsed", args.collapse_dir)]:
        if os.path.exists(condition_dir):
            # We need specific seed dirs for analysis if they exist, or just use the dir
            if os.path.exists(os.path.join(condition_dir, 'checkpoint_50000.pt')):
                t_dir = condition_dir
            else:
                # try to find a seed dir
                seeds = [d for d in os.listdir(condition_dir) if os.path.isdir(os.path.join(condition_dir, d))]
                t_dir = os.path.join(condition_dir, seeds[0]) if seeds else condition_dir

            print(f"Tracking circuit formation for {condition_name} in {t_dir}")
            try:
                traj, steps = track_circuit_formation(t_dir)
                if not traj:
                    print(f"  No checkpoints found in {t_dir}")
                    continue
                print(f"{condition_name} run trajectories:")
                for k, v in traj.items():
                    # Format nicely
                    first_three = [f"{x:.4f}" for x in v[:3]]
                    last_three = [f"{x:.4f}" for x in v[-3:]]
                    print(f"  {k}: [{', '.join(first_three)}] ... [{', '.join(last_three)}]")
            except Exception as e:
                print(f"  Error processing {t_dir}: {e}")
        else:
            print(f"Directory {condition_dir} does not exist. Skipping.")
