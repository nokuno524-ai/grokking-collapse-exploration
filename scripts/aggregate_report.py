import argparse
from pathlib import Path
import markdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np

def load_run_stats(results_dir: Path):
    """Load attention stats from all conditions and runs"""
    all_dfs = []
    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir(): continue
        for run_dir in sorted(condition_dir.iterdir()):
            if not run_dir.is_dir(): continue
            stats_csv = run_dir / "attention_stats.csv"
            if stats_csv.exists():
                df = pd.read_csv(stats_csv)
                df['condition'] = condition_dir.name
                df['run'] = run_dir.name
                all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None

def generate_comparisons(df: pd.DataFrame, out_dir: Path):
    """
    Generate side-by-side comparisons of pure vs medium_collapse vs severe_collapse
    """
    if df is None: return []

    conditions_to_compare = ["pure", "medium_collapse", "severe_collapse"]
    # Check if we have these conditions
    available = df['condition'].unique()
    present = [c for c in conditions_to_compare if c in available]

    if len(present) < 2: return []

    # We'll plot trajectories side-by-side for the first run in each condition as an example
    df['layer_head'] = df['layer'].astype(str) + "_" + df['head'].astype(str)

    fig, axes = plt.subplots(3, len(present), figsize=(5 * len(present), 12))

    if len(present) == 1:
        axes = axes[:, np.newaxis]

    for i, cond in enumerate(present):
        cond_df = df[df['condition'] == cond]
        # just pick the first run
        first_run = cond_df['run'].iloc[0]
        run_df = cond_df[cond_df['run'] == first_run]

        sns.lineplot(data=run_df, x="step", y="entropy", hue="layer_head", ax=axes[0, i])
        axes[0, i].set_title(f"{cond} - Row Entropy")

        sns.lineplot(data=run_df, x="step", y="diag_dominance", hue="layer_head", ax=axes[1, i])
        axes[1, i].set_title(f"{cond} - Diagonal Dominance")

        sns.lineplot(data=run_df, x="step", y="kl_uniform", hue="layer_head", ax=axes[2, i])
        axes[2, i].set_title(f"{cond} - KL Divergence from Uniform")

    plt.tight_layout()
    comparison_img = out_dir / "cross_condition_comparison.png"
    plt.savefig(comparison_img)
    plt.close(fig)
    return [comparison_img]

def generate_html_report(results_dir: Path, out_file: Path):
    """
    Generate an HTML report summarizing attention analysis across multiple runs.
    Expects results_dir to contain subdirectories (one per condition), each containing run dirs.
    """
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_run_stats(results_dir)

    html_content = ["<h1>Attention Analysis Report</h1>"]

    if df is not None:
        comp_imgs = generate_comparisons(df, out_dir)
        if comp_imgs:
            html_content.append("<h2>Cross-Condition Comparisons</h2>")
            for img in comp_imgs:
                with open(img, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                html_content.append(f'<img src="data:image/png;base64,{img_data}" style="max-width:1200px;"><br>')

    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir(): continue
        html_content.append(f"<h2>Condition: {condition_dir.name}</h2>")

        for run_dir in sorted(condition_dir.iterdir()):
            if not run_dir.is_dir(): continue

            # Check if there is a stats CSV here
            stats_csv = run_dir / "attention_stats.csv"
            if not stats_csv.exists(): continue

            html_content.append(f"<h3>Run: {run_dir.name}</h3>")

            # Embed trajectories image
            traj_img = run_dir / "attention_trajectories.png"
            if traj_img.exists():
                with open(traj_img, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                html_content.append(f'<img src="data:image/png;base64,{img_data}" style="max-width:800px;"><br>')

    with open(out_file, "w") as f:
        f.write("\n".join(html_content))

    if df is not None:
        # Also dump the aggregated CSV
        df.to_csv(out_dir / "aggregated_attention_stats.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="Path to base results directory")
    parser.add_argument("--out-file", type=str, required=True, help="Path to output HTML file")
    args = parser.parse_args()

    generate_html_report(Path(args.results_dir), Path(args.out_file))
