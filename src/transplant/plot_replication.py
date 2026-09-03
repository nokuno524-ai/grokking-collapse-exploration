import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.transplant.stats import aggregate_results

def plot_heatmaps(agg_df: pd.DataFrame, out_dir: Path):
    """Plot heatmaps of mean accuracy delta by layer and head."""
    # Filter for head transplants
    head_df = agg_df[agg_df['component_type'] == 'head'].copy()

    if len(head_df) == 0:
        print("No head transplant data found.")
        return

    pairs = head_df[['donor_condition', 'recipient_condition']].drop_duplicates().values

    for donor, recip in pairs:
        subset = head_df[(head_df['donor_condition'] == donor) & (head_df['recipient_condition'] == recip)]

        # Create a pivot table: rows = layer_idx, cols = head_idx, values = mean_acc_delta
        pivot = subset.pivot(index='layer_idx', columns='head_idx', values='mean_acc_delta')

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="RdBu", center=0, ax=ax, fmt=".3f")
        ax.set_title(f"Head Transplant Benefit: {donor} -> {recip}\n(Mean Test Acc Delta)")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")

        fig.tight_layout()
        out_file = out_dir / f"heatmap_{donor}_to_{recip}.png"
        fig.savefig(out_file, dpi=160)
        plt.close(fig)

def generate_markdown_report(agg_df: pd.DataFrame, out_file: Path):
    """Generate a Markdown report summarizing the transplant replication."""
    lines = [
        "# Circuit Transplant Replication Report\n",
        "## Summary of Effects\n",
        "This report aggregates multi-seed circuit transplant experiments, highlighting which components rescue generalization when swapped from a grokked donor to a collapsed recipient.\n\n"
    ]

    pairs = agg_df[['donor_condition', 'recipient_condition']].drop_duplicates().values

    for donor, recip in pairs:
        lines.append(f"### {donor} $\\rightarrow$ {recip}\n")

        subset = agg_df[(agg_df['donor_condition'] == donor) & (agg_df['recipient_condition'] == recip)]

        # Sort by effect size (Cohen's d) descending
        subset = subset.sort_values(by='cohens_d', ascending=False)

        lines.append("| Component | Mean $\\Delta$ Acc | 95% CI | Cohen's $d$ | Const. Attn % |")
        lines.append("|-----------|-------------------|--------|-------------|---------------|")

        for _, row in subset.iterrows():
            comp_name = f"Layer {row['layer_idx']} "
            if row['component_type'] == 'head':
                comp_name += f"Head {int(row['head_idx'])}"
            else:
                comp_name += row['component_type'].upper()

            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"

            lines.append(f"| {comp_name} | {row['mean_acc_delta']:.3f} | {ci_str} | {row['cohens_d']:.2f} | {row['is_constant_attention_frac']*100:.0f}% |")

        lines.append("\n")

    out_file.write_text("\n".join(lines))
    print(f"Saved Markdown report to {out_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, required=True, help="Raw CSV from run_transplants.py")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/transplant"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    # Aggregate stats
    agg_df = aggregate_results(df)
    agg_df.to_csv(args.output_dir / "transplant_aggregated.csv", index=False)
    print(f"Saved aggregated results to {args.output_dir / 'transplant_aggregated.csv'}")

    # Generate plots and report
    plot_heatmaps(agg_df, args.output_dir)
    generate_markdown_report(agg_df, args.output_dir / "transplant_replication.md")

if __name__ == "__main__":
    main()
