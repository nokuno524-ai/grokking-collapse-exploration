"""
Ablation Atlas evaluation and aggregation script.

This script runs systematic component-wise interventions (transplants,
zero-ablation, mean-ablation) on a model, evaluating the effect on
downstream metrics (test accuracy, test loss).

Outputs a CSV and heatmaps.
"""

import argparse
import json
import csv
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from src.model import ModularArithmeticTransformer
from src.transplant_rescue import load_run, build_model, make_loaders, evaluate_model
from src.transplant.atlas import (
    patch_attention_head,
    patch_mlp_neuron,
    patch_layer,
    InterventionType,
    ComponentType,
)

# For statistical analysis added in a later step
import scipy.stats as stats
import statsmodels.stats.multitest as smm


def run_atlas(
    base_run_dir: Path,
    donor_run_dir: Path,
    out_dir: Path,
    base_step: Optional[int] = None,
    donor_step: Optional[int] = None,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
):
    """
    Run systematic component interventions (ablations and transplants) across all layers, heads, and neurons.

    Loads a base (target) model and a donor (reference) model, iterating across components
    to apply `transplant`, `zero`, and `mean` interventions. The model's loss and accuracy
    are recorded, and statistics (CIs, Benjamini-Hochberg corrected p-values) are computed.
    Results are saved as a CSV and visualized via heatmaps.

    Args:
        base_run_dir: Path to the target model run directory (usually contaminated/collapsed).
        donor_run_dir: Path to the reference model run directory (usually pure/healthy).
        out_dir: Path to output directory.
        base_step: Checkpoint step for the target model.
        donor_step: Checkpoint step for the reference model.
        batch_size: Batch size for evaluation.
        device: Torch device.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load runs
    base_sd, base_cfg = load_run(base_run_dir, step=base_step)
    donor_sd, donor_cfg = load_run(donor_run_dir, step=donor_step)

    # Check compatibility
    assert base_cfg.get("seed") == donor_cfg.get("seed"), "Seeds must match for valid test splits"
    assert base_cfg.get("prime") == donor_cfg.get("prime")
    assert base_cfg.get("d_model") == donor_cfg.get("d_model")

    # Extract structural constants
    n_layers = int(base_cfg.get("n_layers", 1))
    n_heads = int(base_cfg.get("n_heads", 4))
    d_model = int(base_cfg.get("d_model", 128))
    d_ff = int(base_cfg.get("d_ff", 512))

    # 2. Get loaders (based on base config)
    train_loader, test_loader = make_loaders(base_cfg, batch_size=batch_size, device=device)

    # 3. Base evaluation (to get baseline accuracy)
    model = build_model(base_cfg, device)
    model.load_state_dict(base_sd)
    model.eval()

    baseline_metrics = evaluate_model(model, train_loader, test_loader, device)
    base_acc = baseline_metrics["test_acc"]

    print(f"Baseline (contam) test acc: {base_acc:.4f}")

    # Interventions to test
    interventions: List[InterventionType] = ["transplant", "zero", "mean"]

    results = []

    # Helper to evaluate a modified state dict
    def eval_sd(mod_sd, component_type, layer_idx, component_idx, intervention):
        model.load_state_dict(mod_sd)
        model.eval()
        with torch.no_grad():
            metrics = evaluate_model(model, train_loader, test_loader, device)

        results.append({
            "component_type": component_type,
            "layer_idx": layer_idx,
            "component_idx": component_idx,
            "intervention": intervention,
            "test_acc": metrics["test_acc"],
            "test_loss": metrics["test_loss"],
            "acc_diff": metrics["test_acc"] - base_acc,
        })

    print("Running layer interventions...")
    for layer_idx in range(n_layers):
        for intervention in interventions:
            mod_sd = patch_layer(base_sd, donor_sd, layer_idx, intervention)
            eval_sd(mod_sd, "layer", layer_idx, 0, intervention)

    print("Running head interventions...")
    for layer_idx in range(n_layers):
        for head_idx in tqdm(range(n_heads), desc=f"L{layer_idx} Heads"):
            for intervention in interventions:
                mod_sd = patch_attention_head(
                    base_sd, donor_sd, layer_idx, head_idx, d_model, n_heads, intervention
                )
                eval_sd(mod_sd, "head", layer_idx, head_idx, intervention)

    print("Running neuron interventions...")
    for layer_idx in range(n_layers):
        # We might not want to log every single neuron in standard output
        for neuron_idx in tqdm(range(d_ff), desc=f"L{layer_idx} Neurons"):
            for intervention in interventions:
                mod_sd = patch_mlp_neuron(
                    base_sd, donor_sd, layer_idx, neuron_idx, intervention
                )
                eval_sd(mod_sd, "neuron", layer_idx, neuron_idx, intervention)

    # 4. Statistical Rigor: Confidence Intervals and Multiple Comparison Correction
    print("Computing statistics...")
    df = pd.DataFrame(results)

    # Bootstrapping (simplified simulation of confidence intervals for effect size)
    # Since we evaluate on the whole test set, the metric itself is stable, but we want
    # to measure if the diff is statistically significant across runs (or simulated noise).
    # We will simulate standard error and p-values based on a theoretical N size (size of test set).
    test_n = len(test_loader.dataset)

    def compute_p_value(acc1, acc2, n):
        # Two-proportion z-test
        # H0: p1 = p2
        # H1: p1 != p2
        p_pool = (acc1 * n + acc2 * n) / (2 * n)
        if p_pool == 0 or p_pool == 1:
            return 1.0 # Cannot reject H0
        se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
        z = (acc2 - acc1) / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        return p_val

    # We calculate the p-value comparing each intervention to the baseline
    df['p_value'] = df.apply(lambda row: compute_p_value(base_acc, row['test_acc'], test_n), axis=1)

    # Benjamini-Hochberg Correction per component type
    df['p_adj'] = 1.0
    for comp_type in df['component_type'].unique():
        mask = df['component_type'] == comp_type
        _, pvals_corrected, _, _ = smm.multipletests(
            df.loc[mask, 'p_value'].values,
            alpha=0.05,
            method='fdr_bh'
        )
        df.loc[mask, 'p_adj'] = pvals_corrected

    df['significant'] = df['p_adj'] < 0.05

    # Bootstrapped CI for effect size (using binomial assumption)
    def bootstrap_ci(acc, n, num_resamples=1000):
        # Generate empirical bootstrap of the accuracy
        successes = int(acc * n)
        samples = np.random.binomial(n, successes / n, num_resamples) / n
        return np.percentile(samples, [2.5, 97.5])

    ci_lower = []
    ci_upper = []
    for idx, row in df.iterrows():
        lower, upper = bootstrap_ci(row['test_acc'], test_n)
        ci_lower.append(lower - base_acc) # CI for the difference
        ci_upper.append(upper - base_acc)

    df['ci_lower'] = ci_lower
    df['ci_upper'] = ci_upper

    # 5. Save results to CSV
    csv_path = out_dir / "atlas_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw results to {csv_path}")

    # 6. Generate Heatmaps
    generate_heatmaps(df, out_dir, n_layers, n_heads, d_ff)

    # 7. Generate Summary Markdown
    generate_summary_markdown(df, out_dir)

def generate_summary_markdown(df: pd.DataFrame, out_dir: Path):
    """Generate a Markdown summary of the significant findings."""
    md_path = out_dir / "MECHANISTIC_SUMMARY.md"

    sig_transplants = df[(df['intervention'] == 'transplant') & (df['significant'] == True) & (df['acc_diff'] > 0)]
    sig_transplants = sig_transplants.sort_values(by='acc_diff', ascending=False)

    sig_ablations = df[(df['intervention'] != 'transplant') & (df['significant'] == True) & (df['acc_diff'] < 0)]
    sig_ablations = sig_ablations.sort_values(by='acc_diff', ascending=True)

    with open(md_path, 'w') as f:
        f.write("# Mechanistic Summary: Ablation and Transplant Atlas\n\n")

        f.write("## Significant Rescue Effects (Transplants)\n")
        f.write("The following components significantly improved performance when transplanted from a pure model to a contaminated model:\n\n")
        if sig_transplants.empty:
            f.write("*No significant rescue effects found.*\n")
        else:
            for idx, row in sig_transplants.head(10).iterrows():
                f.write(f"- **{row['component_type'].capitalize()} L{row['layer_idx']} C{row['component_idx']}**: ")
                f.write(f"Accuracy Diff: +{row['acc_diff']:.4f} ")
                f.write(f"(95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], adj p={row['p_adj']:.2e})\n")

        f.write("\n## Significant Harmful Interventions (Ablations)\n")
        f.write("The following zero/mean ablations significantly degraded the model's performance:\n\n")
        if sig_ablations.empty:
            f.write("*No significant ablations found.*\n")
        else:
            for idx, row in sig_ablations.head(10).iterrows():
                f.write(f"- **{row['intervention'].capitalize()} Ablation {row['component_type'].capitalize()} L{row['layer_idx']} C{row['component_idx']}**: ")
                f.write(f"Accuracy Diff: {row['acc_diff']:.4f} ")
                f.write(f"(95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], adj p={row['p_adj']:.2e})\n")

    print(f"Saved markdown summary to {md_path}")

def generate_heatmaps(df: pd.DataFrame, out_dir: Path, n_layers: int, n_heads: int, d_ff: int):
    """Generate heatmaps of intervention effects."""

    # Plot Head Interventions (Layer vs Head)
    head_df = df[df["component_type"] == "head"]
    if not head_df.empty:
        for intervention in head_df["intervention"].unique():
            sub_df = head_df[head_df["intervention"] == intervention]
            # Pivot table to layer x head
            pivot = sub_df.pivot(index="layer_idx", columns="component_idx", values="acc_diff")

            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, cmap="RdBu", center=0, annot=True, fmt=".3f")
            plt.title(f"Head Intervention: {intervention} (Acc Diff)")
            plt.xlabel("Head Index")
            plt.ylabel("Layer Index")
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_head_{intervention}.png")
            plt.close()

    # Plot Neuron Interventions (Distribution per layer, because there are too many for a clean heatmap)
    neuron_df = df[df["component_type"] == "neuron"]
    if not neuron_df.empty:
        for intervention in neuron_df["intervention"].unique():
            sub_df = neuron_df[neuron_df["intervention"] == intervention]

            plt.figure(figsize=(10, 6))
            sns.violinplot(data=sub_df, x="layer_idx", y="acc_diff")
            plt.title(f"Neuron Intervention: {intervention} (Acc Diff Distribution)")
            plt.xlabel("Layer Index")
            plt.ylabel("Acc Diff")
            plt.axhline(0, color="k", linestyle="--")
            plt.tight_layout()
            plt.savefig(out_dir / f"violin_neuron_{intervention}.png")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ablation Atlas.")
    parser.add_argument("--base-run", type=Path, required=True, help="Path to base (e.g. contam) run")
    parser.add_argument("--donor-run", type=Path, required=True, help="Path to donor (e.g. pure) run")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/atlas"), help="Output directory")
    parser.add_argument("--base-step", type=int, default=None, help="Checkpoint step for base run")
    parser.add_argument("--donor-step", type=int, default=None, help="Checkpoint step for donor run")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    run_atlas(
        args.base_run,
        args.donor_run,
        args.out_dir,
        args.base_step,
        args.donor_step,
        args.batch_size,
        torch.device(args.device)
    )
