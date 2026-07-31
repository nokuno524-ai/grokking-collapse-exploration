import os
import argparse
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob

def generate_sbatch(output_dir: str, python_script: str = "src/train.py"):
    """
    Generate an sbatch array script for a grid search over collapse severity and collapse level (data composition).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Grid definition
    collapse_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] # Data composition ratio
    collapse_severities = [0.1, 0.3, 0.5, 0.7, 0.9] # Collapse severity (temperature warping)
    seeds = [42, 43, 44]

    # Generate job list
    job_file = os.path.join(output_dir, "jobs.txt")
    with open(job_file, 'w') as f:
        for level, severity, seed in itertools.product(collapse_levels, collapse_severities, seeds):
            # Pure condition (level 0) doesn't need all severity permutations
            if level == 0.0 and severity != collapse_severities[0]:
                continue

            out_path = f"results/phase_diagram/level{level}_sev{severity}/seed_{seed}"
            cmd = f"python {python_script} --collapse-level {level} --collapse-severity {severity} --seed {seed} --output-dir {out_path} --max-steps 50000"
            f.write(f"{cmd}\n")

    # Generate sbatch script
    sbatch_script = os.path.join(output_dir, "run_phase_diagram.sbatch")
    num_jobs = sum(1 for _ in open(job_file))

    with open(sbatch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=phase_diagram
#SBATCH --output=slurm/logs/phase_diagram_%A_%a.out
#SBATCH --error=slurm/logs/phase_diagram_%A_%a.err
#SBATCH --array=1-{num_jobs}%20
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.10
source .venv/bin/activate

# Extract command from jobs list
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job_file})
echo "Running: $CMD"
eval $CMD
""")

    print(f"Generated {num_jobs} jobs in {job_file}")
    print(f"Generated sbatch script: {sbatch_script}")
    print(f"Run with: sbatch {sbatch_script}")

def parse_results(results_dir: str = "results/phase_diagram") -> pd.DataFrame:
    """Parse results.json from all grid runs."""
    records = []

    for results_file in glob.glob(f"{results_dir}/**/results.json", recursive=True):
        with open(results_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        # Extract metadata from path or data
        path_parts = Path(results_file).parts

        # Determine grokking step
        test_acc = np.array(data.get('test_acc', []))
        steps = np.array(data.get('step', [i * 100 for i in range(len(test_acc))]))

        grok_step = -1
        if len(test_acc) > 0 and np.max(test_acc) >= 0.9:
            diffs = np.diff(test_acc)
            jump_indices = np.where(diffs > 0.5)[0]
            if len(jump_indices) > 0:
                grok_step = steps[jump_indices[0]]
            else:
                cross_90 = np.where(test_acc >= 0.9)[0]
                if len(cross_90) > 0:
                    grok_step = steps[cross_90[0]]

        records.append({
            'collapse_level': data.get('config', {}).get('collapse_level', 0.0),
            'collapse_severity': data.get('config', {}).get('collapse_severity', 0.0),
            'seed': data.get('config', {}).get('seed', 0),
            'final_test_acc': test_acc[-1] if len(test_acc) > 0 else 0,
            'grok_step': grok_step,
            'grokked': grok_step != -1
        })

    return pd.DataFrame(records)

def plot_phase_diagram(df: pd.DataFrame, output_dir: str = "analysis/phase_diagram"):
    """Generate 2D phase diagrams for grokking outcomes."""
    if df.empty:
        print("No data to plot phase diagram.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Aggregate over seeds
    agg_df = df.groupby(['collapse_level', 'collapse_severity']).agg({
        'grokked': 'mean', # Proportion of seeds that grokked
        'grok_step': lambda x: np.mean(x[x != -1]) if any(x != -1) else np.nan,
        'final_test_acc': 'mean'
    }).reset_index()

    # 1. Plot Grokking Probability Heatmap
    plt.figure(figsize=(10, 8))
    pivot_prob = agg_df.pivot(index='collapse_severity', columns='collapse_level', values='grokked')
    sns.heatmap(pivot_prob, annot=True, cmap='YlGnBu', fmt='.2f', vmin=0, vmax=1)
    plt.title('Grokking Probability (Phase Diagram)')
    plt.ylabel('Collapse Severity (Temperature)')
    plt.xlabel('Collapse Level (Data Fraction)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grokking_probability.png"), dpi=300)
    plt.close()

    # 2. Plot Grokking Step Heatmap (where grokking occurred)
    plt.figure(figsize=(10, 8))
    pivot_step = agg_df.pivot(index='collapse_severity', columns='collapse_level', values='grok_step')
    sns.heatmap(pivot_step, annot=True, cmap='viridis_r', fmt='.0f')
    plt.title('Average Grokking Step (Lower is faster)')
    plt.ylabel('Collapse Severity (Temperature)')
    plt.xlabel('Collapse Level (Data Fraction)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grokking_step.png"), dpi=300)
    plt.close()

    print(f"Saved phase diagram plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="slurm/phase_diagram")
    parser.add_argument("--script", type=str, default="src/train.py")
    parser.add_argument("--analyze", action="store_true", help="Parse results and plot phase diagram")
    parser.add_argument("--results-dir", type=str, default="results/phase_diagram")
    parser.add_argument("--analysis-dir", type=str, default="analysis/phase_diagram")

    args, _ = parser.parse_known_args()

    if args.analyze:
        df = parse_results(args.results_dir)
        plot_phase_diagram(df, args.analysis_dir)
        if not df.empty:
            df.to_csv(os.path.join(args.analysis_dir, "grid_results.csv"), index=False)
    else:
        generate_sbatch(args.output_dir, args.script)
