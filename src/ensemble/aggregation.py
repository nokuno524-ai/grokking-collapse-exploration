"""
Aggregation of multi-seed experiments with statistical rigor.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

try:
    from .utils import load_run_results
    from ..rigorous_stats.stats_module import compute_bootstrap_ci, run_mannwhitneyu
    from ..rigorous_stats.cliff_detection import detect_cliff
except ImportError:
    from utils import load_run_results
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rigorous_stats.stats_module import compute_bootstrap_ci, run_mannwhitneyu
    from rigorous_stats.cliff_detection import detect_cliff

def aggregate_trajectories(results: Dict[str, List[Dict[str, Any]]], metric: str) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate a time-series metric over ragged runs.
    """
    agg = {}
    for condition, runs in results.items():
        # Collect all steps across all runs
        all_steps = set()
        for r in runs:
            history = r.get("history", [])
            for entry in history:
                if entry.get(metric) is not None and not np.isnan(entry.get(metric)):
                    all_steps.add(entry["step"])

        sorted_steps = sorted(list(all_steps))
        if not sorted_steps:
            continue

        step_to_values = {s: [] for s in sorted_steps}

        for r in runs:
            history = r.get("history", [])
            for entry in history:
                val = entry.get(metric)
                if val is not None and not np.isnan(val):
                    step_to_values[entry["step"]].append(val)

        means = []
        stds = []
        lows = []
        highs = []

        for s in sorted_steps:
            vals = step_to_values[s]
            if len(vals) > 0:
                mean, low, high = compute_bootstrap_ci(vals)
                means.append(mean)
                stds.append(np.std(vals))
                lows.append(low)
                highs.append(high)
            else:
                means.append(np.nan)
                stds.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)

        agg[condition] = {
            "steps": sorted_steps,
            "mean": means,
            "std": stds,
            "ci_low": lows,
            "ci_high": highs
        }
    return agg


def aggregate_metrics(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Aggregate metrics across multiple seeds for each condition.

    Args:
        results: Dictionary mapping condition name to list of run results.

    Returns:
        pd.DataFrame containing aggregated metrics (mean, CI, std).
    """
    aggregated_data = []

    for condition, runs in results.items():
        # Extract metrics
        final_accs = [r.get("final_test_acc", np.nan) for r in runs]

        grokking_steps = []
        for r in runs:
            history = r.get("history", [])
            steps = []
            test_accs = []
            for entry in history:
                val = entry.get("test_acc")
                if val is not None and not np.isnan(val):
                    steps.append(entry["step"])
                    test_accs.append(val)

            if len(steps) > 0:
                cliff_res = detect_cliff(steps, test_accs, method='threshold', threshold=0.95)
                if cliff_res is not None:
                    grokking_steps.append(cliff_res["cliff_step"])
                else:
                    grokking_steps.append(np.nan)
            else:
                grokking_steps.append(np.nan)

        final_fourier = [r.get("final_fourier_concentration", np.nan) for r in runs]

        # Calculate CIs and Std
        acc_mean, acc_low, acc_high = compute_bootstrap_ci(final_accs)
        acc_std = np.std([x for x in final_accs if not np.isnan(x)]) if final_accs else np.nan

        grok_mean, grok_low, grok_high = compute_bootstrap_ci(grokking_steps)
        grok_std = np.std([x for x in grokking_steps if not np.isnan(x)]) if grokking_steps else np.nan

        four_mean, four_low, four_high = compute_bootstrap_ci(final_fourier)
        four_std = np.std([x for x in final_fourier if not np.isnan(x)]) if final_fourier else np.nan

        aggregated_data.append({
            "Condition": condition,
            "N": len(runs),
            "Test Acc (Mean ± Std)": f"{acc_mean:.4f} ± {acc_std:.4f}" if not np.isnan(acc_mean) else "NaN",
            "Test Acc (95% CI)": f"[{acc_low:.4f}, {acc_high:.4f}]" if not np.isnan(acc_low) else "NaN",
            "Grok Step (Mean ± Std)": f"{grok_mean:.1f} ± {grok_std:.1f}" if not np.isnan(grok_mean) else "NaN",
            "Grok Step (95% CI)": f"[{grok_low:.1f}, {grok_high:.1f}]" if not np.isnan(grok_low) else "NaN",
            "Fourier (Mean ± Std)": f"{four_mean:.4f} ± {four_std:.4f}" if not np.isnan(four_mean) else "NaN",
            "Fourier (95% CI)": f"[{four_low:.4f}, {four_high:.4f}]" if not np.isnan(four_low) else "NaN"
        })

    return pd.DataFrame(aggregated_data)

def compute_significance(results: Dict[str, List[Dict[str, Any]]], baseline_condition="pure"):
    """
    Compute statistical significance between conditions against a baseline.
    """
    if baseline_condition not in results:
        print(f"Baseline condition '{baseline_condition}' not found in results.")
        return pd.DataFrame()

    baseline_runs = results[baseline_condition]
    baseline_accs = [r.get("final_test_acc", np.nan) for r in baseline_runs]

    sig_data = []
    for condition, runs in results.items():
        if condition == baseline_condition:
            continue

        accs = [r.get("final_test_acc", np.nan) for r in runs]
        test_result = run_mannwhitneyu(baseline_accs, accs)

        sig_data.append({
            "Condition": condition,
            "Comparison": f"{condition} vs {baseline_condition}",
            "MW U-Stat": test_result['statistic'],
            "p-value": test_result['pvalue'],
            "Significant (p<0.05)": test_result['pvalue'] < 0.05 if not np.isnan(test_result['pvalue']) else False
        })

    return pd.DataFrame(sig_data)

def generate_reports(results_dir: str | Path, output_dir: str | Path = None):
    """
    Generate markdown and CSV reports from results directory.
    """
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir.parent / "analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_run_results(results_dir)
    if not results:
        print("No results found.")
        return

    df_agg = aggregate_metrics(results)
    df_sig = compute_significance(results)

    weight_norm_traj = aggregate_trajectories(results, "weight_norm")

    # Flatten trajectory dictionary for CSV export
    traj_data = []
    for cond, data in weight_norm_traj.items():
        for i, step in enumerate(data["steps"]):
            traj_data.append({
                "Condition": cond,
                "Step": step,
                "Weight Norm (Mean)": data["mean"][i],
                "Weight Norm (Std)": data["std"][i],
                "Weight Norm (95% CI Low)": data["ci_low"][i],
                "Weight Norm (95% CI High)": data["ci_high"][i]
            })
    df_traj = pd.DataFrame(traj_data)

    # Save CSVs
    df_agg.to_csv(output_dir / "aggregated_metrics.csv", index=False)
    if not df_sig.empty:
        df_sig.to_csv(output_dir / "significance_tests.csv", index=False)
    if not df_traj.empty:
        df_traj.to_csv(output_dir / "weight_norm_trajectories.csv", index=False)

    # Generate Markdown
    md_content = "# Experimental Results Summary\n\n"
    md_content += "## Aggregated Metrics\n\n"
    md_content += df_agg.to_markdown(index=False) + "\n\n"

    if not df_sig.empty:
        md_content += "## Statistical Significance (Mann-Whitney U)\n\n"
        md_content += df_sig.to_markdown(index=False) + "\n\n"

    if not df_traj.empty:
        md_content += "## Weight Norm Trajectories\n\n"
        md_content += "*Time-series trajectories exported to `weight_norm_trajectories.csv`.*\n"

    with open(output_dir / "summary.md", "w") as f:
        f.write(md_content)

    print(f"Reports saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="Path to multi-seed results directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()
    generate_reports(args.results_dir, args.output_dir)
