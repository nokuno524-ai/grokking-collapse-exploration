import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Any

from .statistics import compute_cohens_d, t_test_independent
from .utils import _ordered_condition_dirs

def identify_grokking_point(history: List[Dict[str, float]], threshold: float = 0.95) -> int:
    for entry in history:
        if entry.get("test_acc", 0) >= threshold:
            return entry["step"]
    return -1

def run_comprehensive_analysis(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    grokking_steps = []
    final_weight_norms = []
    final_test_accs = []

    pure_accs = []
    collapse_accs = []

    for cond_dir in _ordered_condition_dirs(results_dir):
        if not cond_dir.is_dir() or not (cond_dir / "results.json").exists():
            continue

        with open(cond_dir / "results.json") as f:
            data = json.load(f)

        conditions.append(cond_dir.name)

        # Recalculate grokking point just in case
        history = data.get("history", [])
        g_step = data.get("grokking_step")
        if g_step is None:
            g_step = identify_grokking_point(history)

        grokking_steps.append(g_step if g_step > 0 else 50000)
        final_weight_norms.append(data.get("final_weight_norm", 0))
        test_acc = data.get("final_test_acc", 0)
        final_test_accs.append(test_acc)

        if "pure" in cond_dir.name.lower():
            pure_accs.append(test_acc)
        elif "collapse" in cond_dir.name.lower():
            collapse_accs.append(test_acc)

    # 1. Compare grokking dynamics
    with open(output_dir / "grokking_dynamics_summary.txt", "w") as f:
        f.write("Grokking Dynamics across Collapse Levels:\n")
        for cond, g_step, acc in zip(conditions, grokking_steps, final_test_accs):
            status = "Grokked" if acc >= 0.95 else "Failed"
            f.write(f"- {cond}: {status} (Step: {g_step}, Final Acc: {acc:.4f})\n")

    # 2. Statistical significance testing
    if pure_accs and collapse_accs:
        t_stat, p_val = t_test_independent(np.array(pure_accs), np.array(collapse_accs))
        d_stat = compute_cohens_d(np.array(pure_accs), np.array(collapse_accs))
        with open(output_dir / "statistical_tests.txt", "w") as f:
            f.write(f"Pure vs Collapse Accuracies T-test:\n")
            f.write(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4e}\n")
            f.write(f"Cohen's d: {d_stat:.4f}\n")

    # 3. Correlate weight norm changes
    if len(final_weight_norms) > 1:
        corr, p_corr = pearsonr(final_weight_norms, final_test_accs)
        with open(output_dir / "weight_norm_correlation.txt", "w") as f:
            f.write(f"Correlation (Weight Norm vs Test Acc): {corr:.4f} (p={p_corr:.4e})\n")

    # Generate visualization for Correlation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=final_weight_norms, y=final_test_accs, hue=conditions, s=100)
    plt.title("Weight Norm vs Final Test Accuracy")
    plt.xlabel("Final Weight Norm")
    plt.ylabel("Final Test Accuracy")
    plt.savefig(output_dir / "weight_norm_correlation.png", bbox_inches='tight')
    plt.savefig(output_dir / "weight_norm_correlation.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/comprehensive"))
    args = parser.parse_args()
    run_comprehensive_analysis(args.results_dir, args.output_dir)
