import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.checkpoint import load_run
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from torch.utils.data import TensorDataset, DataLoader
from src.train import evaluate
from src.transplant.alignment import align_models

def get_dataloaders(cfg: Dict, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """
    Construct deterministically seeded dataloaders to maintain exact evaluation alignment.
    Args:
        cfg: Configuration dictionary containing dataset parameters.
        device: Torch device (unused, kept for signature consistency).
    Returns:
        train_loader: Training DataLoader.
        test_loader: Evaluation DataLoader.
    """
    dc = DatasetConfig(
        prime=int(cfg.get("prime", 59)),
        train_fraction=float(cfg.get("train_fraction", 0.3)),
        collapse_level=float(cfg.get("collapse_level", 0.0)),
        collapse_severity=float(cfg.get("collapse_severity", 0.5)),
        noise_fraction=float(cfg.get("noise_fraction", 0.0)),
        seed=int(cfg.get("seed", 42)),
    )
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(dc)
    train_ds = TensorDataset(train_in, train_tgt)
    test_ds = TensorDataset(test_in, test_tgt)
    g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    return train_loader, test_loader

def build_model(cfg: Dict, device: torch.device) -> ModularArithmeticTransformer:
    """
    Instantiate the ModularArithmeticTransformer according to the provided configuration.
    Args:
        cfg: Configuration dictionary containing model architecture parameters.
        device: Torch device to load the model on.
    Returns:
        An uninitialized ModularArithmeticTransformer instance.
    """
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)

def patch_head(base_sd: Dict[str, torch.Tensor], donor_sd: Dict[str, torch.Tensor], layer_idx: int, head_idx: int, d_model: int, n_heads: int) -> Dict[str, torch.Tensor]:
    """
    Surgically replace a single attention head's weights (Q, K, V, and out_proj) from a donor state dict into a base state dict.
    Args:
        base_sd: The foundational state dictionary receiving the patch.
        donor_sd: The state dictionary donating the attention head.
        layer_idx: The index of the transformer layer to modify.
        head_idx: The index of the attention head to replace.
        d_model: The full dimensionality of the model.
        n_heads: Total number of attention heads.
    Returns:
        A new state dictionary with the patched attention head.
    """
    sd = copy.deepcopy(base_sd)
    head_dim = d_model // n_heads

    in_base = sd[f'transformer.layers.{layer_idx}.self_attn.in_proj_weight']
    in_donor = donor_sd[f'transformer.layers.{layer_idx}.self_attn.in_proj_weight']

    # Q, K, V
    q_b, k_b, v_b = in_base.chunk(3, dim=0)
    q_d, k_d, v_d = in_donor.chunk(3, dim=0)

    def patch_qkv(base_w, donor_w):
        b = base_w.view(n_heads, head_dim, d_model)
        d = donor_w.view(n_heads, head_dim, d_model)
        b[head_idx] = d[head_idx]
        return b.view(d_model, d_model)

    sd[f'transformer.layers.{layer_idx}.self_attn.in_proj_weight'] = torch.cat([
        patch_qkv(q_b, q_d), patch_qkv(k_b, k_d), patch_qkv(v_b, v_d)
    ], dim=0)

    if f'transformer.layers.{layer_idx}.self_attn.in_proj_bias' in sd:
        b_base = sd[f'transformer.layers.{layer_idx}.self_attn.in_proj_bias']
        b_donor = donor_sd[f'transformer.layers.{layer_idx}.self_attn.in_proj_bias']
        q_bb, k_bb, v_bb = b_base.chunk(3, dim=0)
        q_dd, k_dd, v_dd = b_donor.chunk(3, dim=0)

        def patch_bias(base_b, donor_b):
            b = base_b.view(n_heads, head_dim)
            d = donor_b.view(n_heads, head_dim)
            b[head_idx] = d[head_idx]
            return b.view(-1)

        sd[f'transformer.layers.{layer_idx}.self_attn.in_proj_bias'] = torch.cat([
            patch_bias(q_bb, q_dd), patch_bias(k_bb, k_dd), patch_bias(v_bb, v_dd)
        ], dim=0)

    # out proj
    out_b = sd[f'transformer.layers.{layer_idx}.self_attn.out_proj.weight']
    out_d = donor_sd[f'transformer.layers.{layer_idx}.self_attn.out_proj.weight']
    ob = out_b.view(d_model, n_heads, head_dim)
    od = out_d.view(d_model, n_heads, head_dim)
    ob[:, head_idx, :] = od[:, head_idx, :]
    sd[f'transformer.layers.{layer_idx}.self_attn.out_proj.weight'] = ob.view(d_model, d_model)

    return sd

def patch_component(base_sd: Dict[str, torch.Tensor], donor_sd: Dict[str, torch.Tensor], component_name: str) -> Dict[str, torch.Tensor]:
    """
    Replace a generic component (e.g. MLP block, Token embedding) from the donor into the base model.
    Args:
        base_sd: Foundational state dict.
        donor_sd: Donating state dict.
        component_name: The prefix name of the component (e.g. 'transformer.layers.0.linear1').
    Returns:
        A new state dictionary with the patched component.
    """
    sd = copy.deepcopy(base_sd)
    for k in sd.keys():
        if k.startswith(component_name):
            sd[k] = donor_sd[k].clone()
    return sd

def eval_sd(sd: Dict[str, torch.Tensor], cfg: Dict, device: torch.device) -> float:
    """
    Load a state dictionary into a model and evaluate test accuracy.
    Args:
        sd: The state dictionary to evaluate.
        cfg: The configuration used to build the corresponding model and dataloaders.
        device: Evaluation device.
    Returns:
        The test accuracy.
    """
    model = build_model(cfg, device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    train_loader, test_loader = get_dataloaders(cfg, device)
    _, test_acc = evaluate(model, test_loader, device)
    return test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-run", type=Path, required=True)
    parser.add_argument("--contam-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/atlas"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    pure_sd, pure_cfg = load_run(args.pure_run)
    contam_sd, contam_cfg = load_run(args.contam_run)

    pure_model = build_model(pure_cfg, device)
    pure_model.load_state_dict(pure_sd, strict=True)
    contam_model = build_model(contam_cfg, device)
    contam_model.load_state_dict(contam_sd, strict=True)

    # Align contam model to pure model
    print("Aligning models...")
    aligned_contam_model, sim_before, sim_after = align_models(pure_model, contam_model)
    print(f"Alignment similarity before: {sim_before:.3f}, after: {sim_after:.3f}")

    aligned_contam_sd = aligned_contam_model.state_dict()

    d_model = pure_cfg.get("d_model", 128)
    n_heads = pure_cfg.get("n_heads", 4)
    n_layers = pure_cfg.get("n_layers", 1)

    # Baseline evaluations
    print("Evaluating baselines...")
    pure_acc = eval_sd(pure_sd, pure_cfg, device)
    contam_acc = eval_sd(aligned_contam_sd, contam_cfg, device)
    print(f"Pure test acc: {pure_acc:.3f}, Contam test acc: {contam_acc:.3f}")

    results = []

    # We test two directions: pure->contam and contam->pure
    directions = [
        ("pure->contam", aligned_contam_sd, pure_sd, contam_cfg, contam_acc),
        ("contam->pure", pure_sd, aligned_contam_sd, pure_cfg, pure_acc)
    ]

    components_to_test = [
        "token_embed",
        "pos_embed",
        "output_head",
        "ln"
    ]

    for i in range(n_layers):
        components_to_test.extend([
            f"transformer.layers.{i}.linear1",
            f"transformer.layers.{i}.linear2",
            f"transformer.layers.{i}.norm1",
            f"transformer.layers.{i}.norm2",
        ])

    for direction_name, base_sd, donor_sd, eval_cfg, baseline_acc in directions:
        for comp in components_to_test:
            print(f"Patching {comp} ({direction_name})")
            patched_sd = patch_component(base_sd, donor_sd, comp)
            acc = eval_sd(patched_sd, eval_cfg, device)
            effect_size = acc - baseline_acc
            results.append({
                "direction": direction_name,
                "component": comp,
                "test_acc": acc,
                "effect_size": effect_size
            })

        # Individual attention heads
        for layer in range(n_layers):
            for head in range(n_heads):
                comp = f"layer_{layer}_head_{head}"
                print(f"Patching {comp} ({direction_name})")
                patched_sd = patch_head(base_sd, donor_sd, layer, head, d_model, n_heads)
                acc = eval_sd(patched_sd, eval_cfg, device)
                effect_size = acc - baseline_acc
                results.append({
                    "direction": direction_name,
                    "component": comp,
                    "test_acc": acc,
                    "effect_size": effect_size
                })

    df = pd.DataFrame(results)
    csv_path = args.output_dir / "atlas_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Plot heatmaps
    for direction in ["pure->contam", "contam->pure"]:
        sub_df = df[df["direction"] == direction]

        fig, ax = plt.subplots(figsize=(10, 6))

        comps = sub_df["component"].values
        effects = sub_df["effect_size"].values

        # sort by effect size
        sorted_indices = np.argsort(effects)
        comps = comps[sorted_indices]
        effects = effects[sorted_indices]

        colors = ['red' if e < 0 else 'green' for e in effects]

        ax.barh(comps, effects, color=colors)
        ax.set_xlabel("Effect Size (Change in Test Acc)")
        ax.set_title(f"Transplant Effect Size: {direction}")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_path = args.output_dir / f"atlas_{direction.replace('->', '_to_')}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
