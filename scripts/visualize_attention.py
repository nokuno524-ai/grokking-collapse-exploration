#!/usr/bin/env python3
"""
CLI tool for extracting and visualizing attention patterns from model checkpoints.
Generates heatmaps, metrics CSVs, and a markdown gallery.
"""

import argparse
import os
import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.model import ModularArithmeticTransformer
from src.analysis.attention import extract_attention_weights, analyze_attention
from src.viz.attention import plot_attention_grid, plot_attention_diff_grid

def get_dummy_batch(prime: int = 59, batch_size: int = 128, seed: int = 42) -> torch.Tensor:
    """Generate a fixed random batch of inputs for consistent evaluation."""
    rng = torch.Generator().manual_seed(seed)
    # Inputs are pairs of tokens in [0, prime)
    return torch.randint(0, prime, (batch_size, 2), generator=rng)

def load_checkpoint(ckpt_path: Path) -> tuple[ModularArithmeticTransformer, dict]:
    """Load model and config from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})

    # Defaults in case config is missing/old
    prime = config.get("prime", 59)
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    d_ff = config.get("d_ff", 512)
    n_layers = config.get("n_layers", 1)

    model = ModularArithmeticTransformer(
        prime=prime, d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers
    )

    # Load state dict strictly to prevent silent failures
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return model, config

def generate_markdown_gallery(
    output_dir: Path,
    single_ckpt_reports: list[dict],
    diff_reports: list[dict]
):
    """Generate a Markdown report embedding all generated figures and tables."""
    lines = ["# Attention Analysis Gallery\n"]

    if single_ckpt_reports:
        lines.append("## Checkpoint Analysis\n")
        for rep in single_ckpt_reports:
            lines.append(f"### {rep['name']}\n")
            lines.append(f"![Attention Heatmap]({rep['heatmap_path'].name})\n")
            lines.append("\n**Metrics Summary:**\n")

            # Show the CSV data as a markdown table
            df = pd.read_csv(rep['csv_path'])
            lines.append(df.to_markdown(index=False))
            lines.append("\n")

    if diff_reports:
        lines.append("## Comparisons (A - B)\n")
        for rep in diff_reports:
            lines.append(f"### {rep['name_a']} vs {rep['name_b']}\n")
            lines.append(f"![Difference Heatmap]({rep['diff_path'].name})\n")
            lines.append("\n")

    md_path = output_dir / "attention_gallery.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Gallery written to {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize attention patterns.")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to checkpoint .pt files")
    parser.add_argument("--names", type=str, nargs="+",
                        help="Display names for the checkpoints (must match length of --checkpoints)")
    parser.add_argument("--compare", action="store_true",
                        help="If true, generates pairwise difference plots for the provided checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/attention_viz",
                        help="Directory to save plots and reports")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for extracting attention")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.checkpoints):
        parser.error("--names must have the same number of arguments as --checkpoints")

    names = args.names or [Path(p).stem for p in args.checkpoints]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store extracted weights for comparisons
    attn_weights = {}
    single_ckpt_reports = []

    # 1. Process individual checkpoints
    for path_str, name in zip(args.checkpoints, names):
        print(f"Processing {name} from {path_str}...")
        ckpt_path = Path(path_str)
        model, config = load_checkpoint(ckpt_path)

        prime = config.get("prime", 59)
        inputs = get_dummy_batch(prime=prime, batch_size=args.batch_size)

        weights = extract_attention_weights(model, inputs)
        attn_weights[name] = weights

        # Analyze metrics
        metrics = analyze_attention(weights)

        # Flatten metrics to a DataFrame for CSV
        # Shape of mean_entropy is (n_layers, n_heads)
        n_layers, n_heads = metrics['mean_entropy'].shape
        rows = []
        for l in range(n_layers):
            for h in range(n_heads):
                rows.append({
                    "layer": l,
                    "head": h,
                    "mean_entropy": metrics['mean_entropy'][l, h],
                    "mean_concentration": metrics['mean_concentration'][l, h]
                })
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{name}_metrics.csv"
        df.to_csv(csv_path, index=False)

        # Also save head similarity matrix
        sim_path = output_dir / f"{name}_similarity.csv"
        pd.DataFrame(metrics['head_similarity']).to_csv(sim_path, index=False)

        # Plot single heatmap
        heatmap_path = output_dir / f"{name}_attention.png"
        plot_attention_grid(weights, output_path=heatmap_path, title=f"Attention: {name}")

        single_ckpt_reports.append({
            "name": name,
            "heatmap_path": heatmap_path,
            "csv_path": csv_path,
        })

    # 2. Perform pairwise comparisons if requested
    diff_reports = []
    if args.compare and len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a = names[i]
                name_b = names[j]
                print(f"Comparing {name_a} vs {name_b}...")

                diff_path = output_dir / f"diff_{name_a}_vs_{name_b}.png"
                plot_attention_diff_grid(
                    attn_weights[name_a],
                    attn_weights[name_b],
                    output_path=diff_path,
                    title=f"Attention Diff: {name_a} - {name_b}"
                )

                diff_reports.append({
                    "name_a": name_a,
                    "name_b": name_b,
                    "diff_path": diff_path
                })

    # 3. Generate Markdown Gallery
    print("Generating gallery...")
    generate_markdown_gallery(output_dir, single_ckpt_reports, diff_reports)
    print("Done!")

if __name__ == "__main__":
    main()
