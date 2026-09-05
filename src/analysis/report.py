"""
Generates multi-seed analysis reports and plots.
"""

import os
import json
import csv
from pathlib import Path
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.analysis.grok_detector.stats import (
    detect_grokking_step,
    bootstrap_median_ci,
    kaplan_meier_survival,
    log_rank_test_multi
)

CONDITION_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

COLORS = {
    "pure": "#2ecc71",
    "low_collapse": "#3498db",
    "medium_collapse": "#f39c12",
    "high_collapse": "#e74c3c",
    "severe_collapse": "#8e44ad",
}

def load_results_csv(filepath: str):
    """Load results from tidy CSV and reconstruct trajectories."""
    runs = defaultdict(lambda: defaultdict(list))

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"]
            seed = int(row["seed"])

            try:
                step = int(row["step"])
                test_acc = float(row["test_acc"])
                wn = float(row["weight_norm"]) if row["weight_norm"] and row["weight_norm"] != "None" else np.nan
            except ValueError:
                continue

            # Dictionary key is (condition, seed)
            runs[(cond, seed)]["steps"].append(step)
            runs[(cond, seed)]["test_accs"].append(test_acc)
            runs[(cond, seed)]["weight_norms"].append(wn)
            runs[(cond, seed)]["severity"] = float(row["collapse_severity"])

    # Convert to structured list format
    structured_runs = {c: [] for c in CONDITION_ORDER}
    for (cond, seed), data in runs.items():
        if cond in structured_runs:
            structured_runs[cond].append({
                "seed": seed,
                "severity": data["severity"],
                "steps": np.array(data["steps"]),
                "test_accs": np.array(data["test_accs"]),
                "weight_norms": np.array(data["weight_norms"]),
            })

    # Sort by step inside each run
    for cond, run_list in structured_runs.items():
        for r in run_list:
            idx = np.argsort(r["steps"])
            r["steps"] = r["steps"][idx]
            r["test_accs"] = r["test_accs"][idx]
            r["weight_norms"] = r["weight_norms"][idx]

    return structured_runs

def generate_report(results_dir: str, output_dir: str):
    """Generate Markdown report and plots."""
    results_path = Path(results_dir) / "results.csv"
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_data = load_results_csv(str(results_path))

    max_step_global = 0

    # Process each condition
    processed_stats = {}
    groups_for_survival = {}

    for cond in CONDITION_ORDER:
        runs = cond_data[cond]
        if not runs:
            continue

        grok_steps = []
        censored = []
        final_norms = []
        severities = []

        for run in runs:
            steps = run["steps"]
            test_acc = run["test_accs"]
            norms = run["weight_norms"]

            if len(steps) == 0:
                continue

            if steps[-1] > max_step_global:
                max_step_global = steps[-1]

            final_norms.append(norms[-1])
            severities.append(run["severity"])

            g_step = detect_grokking_step(steps, test_acc)

            # Save for plotting
            run["detected_grok_step"] = g_step

            if g_step is not None:
                grok_steps.append(g_step)
                censored.append(False)
            else:
                grok_steps.append(steps[-1])
                censored.append(True)

        grok_steps = np.array(grok_steps)
        censored = np.array(censored)
        final_norms = np.array(final_norms)

        # True grok steps for median
        true_grok = grok_steps[~censored]
        if len(true_grok) > 0:
            median, lci, uci = bootstrap_median_ci(true_grok)
        else:
            median, lci, uci = np.nan, np.nan, np.nan

        processed_stats[cond] = {
            "n_runs": len(runs),
            "n_grokked": np.sum(~censored),
            "median_grok": median,
            "grok_lci": lci,
            "grok_uci": uci,
            "final_norm_mean": np.nanmean(final_norms),
            "final_norm_std": np.nanstd(final_norms),
            "final_norm_vals": final_norms,
            "severity": severities[0] if severities else np.nan,
        }

        groups_for_survival[cond] = (grok_steps, censored)

    # Stats tests
    p_values = log_rank_test_multi(groups_for_survival, method="bonferroni")

    # Plotting
    if HAS_MATPLOTLIB:
        # 1. Survival Plot
        plt.figure(figsize=(10, 6))
        for cond in CONDITION_ORDER:
            if cond not in groups_for_survival:
                continue
            times, cens = groups_for_survival[cond]
            utimes, surv, lci, uci = kaplan_meier_survival(times, cens)
            if len(utimes) > 0:
                plt.step(utimes, surv, where='post', label=cond, color=COLORS.get(cond, "gray"))
                plt.fill_between(utimes, lci, uci, step='post', alpha=0.2, color=COLORS.get(cond, "gray"))

        plt.xlabel("Training Step")
        plt.ylabel("Probability of NOT Grokking")
        plt.title("Kaplan-Meier Survival Estimates for Grokking")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "survival_curves.png", dpi=150)
        plt.close()

        # 2. Spaghetti Plot
        plt.figure(figsize=(15, 10))
        for i, cond in enumerate(CONDITION_ORDER):
            if cond not in cond_data or not cond_data[cond]:
                continue
            plt.subplot(2, 3, i + 1)
            plt.title(cond)

            for run in cond_data[cond]:
                steps = run["steps"]
                accs = run["test_accs"]
                plt.plot(steps, accs, alpha=0.3, color=COLORS.get(cond, "gray"))

                # Highlight cliff
                g_step = run.get("detected_grok_step")
                if g_step is not None:
                    idx = np.where(steps == g_step)[0]
                    if len(idx) > 0:
                        plt.scatter(g_step, accs[idx[0]], color='red', zorder=5, s=20)

            plt.axhline(0.95, color='r', linestyle='--', alpha=0.5)
            plt.ylim(0, 1.05)
            if max_step_global > 0:
                plt.xlim(0, max_step_global)
            plt.xlabel("Step")
            plt.ylabel("Test Acc")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "spaghetti_plots.png", dpi=150)
        plt.close()

        # 3. Weight Norm vs Severity with CIs
        plt.figure(figsize=(8, 6))
        severities = []
        means = []
        errs = []

        for cond in CONDITION_ORDER:
            if cond in processed_stats:
                st = processed_stats[cond]
                severities.append(st["severity"])
                means.append(st["final_norm_mean"])
                # 95% CI roughly 1.96 * std / sqrt(n)
                n = len(st["final_norm_vals"][~np.isnan(st["final_norm_vals"])])
                if n > 0:
                    se = st["final_norm_std"] / np.sqrt(n)
                    errs.append(1.96 * se)
                else:
                    errs.append(0)

        # Sort by severity
        idx = np.argsort(severities)
        severities = np.array(severities)[idx]
        means = np.array(means)[idx]
        errs = np.array(errs)[idx]

        plt.errorbar(severities, means, yerr=errs, marker='o', capsize=5, linestyle='-', color='indigo')
        plt.xlabel("Collapse Severity")
        plt.ylabel("Final Weight Norm")
        plt.title("Weight Norm Reduction Correlates with Severity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "weight_norm_vs_severity.png", dpi=150)
        plt.close()

    # Generate Markdown
    md_lines = []
    md_lines.append("# Multi-Seed Grokking Analysis Report")
    md_lines.append("")
    md_lines.append("## Summary Statistics")
    md_lines.append("")
    md_lines.append("| Condition | Runs | Grokked | Median Grok Step [95% CI] | Final Weight Norm |")
    md_lines.append("|---|---|---|---|---|")

    for cond in CONDITION_ORDER:
        if cond not in processed_stats:
            continue
        st = processed_stats[cond]

        if np.isnan(st["median_grok"]):
            med_str = "Never"
        else:
            med_str = f"{st['median_grok']:.0f} [{st['grok_lci']:.0f}, {st['grok_uci']:.0f}]"

        norm_str = f"{st['final_norm_mean']:.2f} ± {st['final_norm_std']:.2f}"

        md_lines.append(f"| {cond} | {st['n_runs']} | {st['n_grokked']} | {med_str} | {norm_str} |")

    md_lines.append("")
    md_lines.append("## Log-Rank Test p-values (Bonferroni Corrected)")
    md_lines.append("")

    # Create matrix
    md_lines.append("| Condition | " + " | ".join(CONDITION_ORDER) + " |")
    md_lines.append("|---|" + "|".join(["---"] * len(CONDITION_ORDER)) + "|")

    for c1 in CONDITION_ORDER:
        row = [c1]
        for c2 in CONDITION_ORDER:
            if c1 == c2:
                row.append("-")
            else:
                pair1 = (c1, c2)
                pair2 = (c2, c1)
                pval = p_values.get(pair1, p_values.get(pair2, np.nan))
                if np.isnan(pval):
                    row.append("N/A")
                elif pval < 0.001:
                    row.append("<0.001**")
                elif pval < 0.01:
                    row.append(f"{pval:.3f}*")
                else:
                    row.append(f"{pval:.3f}")
        md_lines.append("| " + " | ".join(row) + " |")

    if HAS_MATPLOTLIB:
        md_lines.append("")
        md_lines.append("## Visualizations")
        md_lines.append("")
        md_lines.append("### Survival Estimates")
        md_lines.append("![Survival Curves](survival_curves.png)")
        md_lines.append("")
        md_lines.append("### Test Accuracy Trajectories")
        md_lines.append("![Spaghetti Plots](spaghetti_plots.png)")
        md_lines.append("")
        md_lines.append("### Weight Norm vs Severity")
        md_lines.append("![Weight Norm](weight_norm_vs_severity.png)")
        md_lines.append("")

    with open(out_dir / "grokking_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Report saved to {out_dir / 'grokking_report.md'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/multi_seed")
    parser.add_argument("--output-dir", type=str, default="analysis/multi_seed")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output_dir)
