import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.grok_detector.run_aggregator import aggregate_condition_runs, extract_metric, load_run_history
from src.analysis.grok_detector.stats import kaplan_meier_estimator, log_rank_test, permutation_test_median
from src.analysis.grok_detector.detectors import piecewise_linear_detector

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

def plot_survival_curves(km_results: Dict[str, dict], output_path: Path):
    plt.figure(figsize=(10, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    for cond in SEVERITY_ORDER:
        if cond not in km_results:
            continue

        res = km_results[cond]
        t = res["times"]
        s = res["survival"]
        lower = res["lower"]
        upper = res["upper"]

        c = colors.get(cond, "black")

        # Plot step function for Kaplan-Meier
        plt.step(t, s, where='post', label=cond, color=c, linewidth=2)
        plt.fill_between(t, lower, upper, step='post', color=c, alpha=0.2)

    plt.title("Grokking Survival Analysis (Time to Grok)")
    plt.xlabel("Training Step")
    plt.ylabel("Probability of Not Grokking")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_grok_step_distributions(agg_results: Dict[str, dict], output_path: Path):
    plt.figure(figsize=(10, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    labels = []
    data = []
    box_colors = []

    for cond in SEVERITY_ORDER:
        if cond not in agg_results:
            continue

        steps = [r["grok_step"] for r in agg_results[cond]["runs"] if r["grokked"] and r["grok_step"] is not None]
        if steps:
            data.append(steps)
            labels.append(cond)
            box_colors.append(colors.get(cond, "black"))

    if not data:
        plt.close()
        return

    bplot = plt.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title("Grokking Step Distributions")
    plt.ylabel("Training Step")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_analysis(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_results = {}
    km_results = {}
    method_comparisons = []

    print("Aggregating runs and computing survival estimators...")
    from src.analysis.grok_detector.detectors import (
        piecewise_linear_detector, smoothed_derivative_detector, bocd_detector
    )

    for cond in SEVERITY_ORDER:
        cond_dir = results_dir / cond

        # In our project structure, condition_dir is conceptually the root of the condition
        # run_aggregator handles finding the seed_dirs from it
        agg = aggregate_condition_runs(cond_dir, calc_ci=True)
        if agg["stats"]["num_runs"] == 0:
            continue

        agg_results[cond] = agg

        # Prepare data for survival analysis
        durations = []
        events = []
        for r in agg["runs"]:
            if r["grokked"]:
                durations.append(r["grok_step"])
                events.append(1)
            else:
                durations.append(r["max_step"])
                events.append(0)

        if durations:
            dur = np.array(durations)
            ev = np.array(events)
            t, s, lower, upper = kaplan_meier_estimator(dur, ev)
            km_results[cond] = {
                "durations": dur,
                "events": ev,
                "times": t,
                "survival": s,
                "lower": lower,
                "upper": upper
            }

        if cond == "pure":
            # Just do one manual extraction for method comparison
            for seed_dir in [d for d in results_dir.iterdir() if d.is_dir()]:
                actual_cond = seed_dir / cond
                history = load_run_history(actual_cond)
                if history:
                    steps, values = extract_metric(history, 'test_acc')
                    if len(steps) >= 4:
                        cp_pw = piecewise_linear_detector(steps, values)
                        cp_sd = smoothed_derivative_detector(steps, values)
                        cp_bocd = bocd_detector(steps, values, mean0=0.5, var0=1.0, varx=0.01)
                        method_comparisons.append({
                            "seed": seed_dir.name,
                            "piecewise": cp_pw,
                            "smoothed_deriv": cp_sd,
                            "bocd": cp_bocd
                        })
                        break

    # Plots
    plot_survival_curves(km_results, output_dir / "survival_curves.png")
    plot_grok_step_distributions(agg_results, output_dir / "grok_step_distributions.png")

    # Statistical tests
    stats_markdown = []

    # Pairwise Log-rank and Permutation tests
    available_conds = [c for c in SEVERITY_ORDER if c in km_results]

    stats_markdown.append("### Pairwise Statistical Comparisons\n")
    stats_markdown.append("| Comparison | Log-Rank p-value | Permutation p-value (Median) | Significant? (p<0.05) |")
    stats_markdown.append("|------------|------------------|------------------------------|-----------------------|")

    for i in range(len(available_conds)):
        for j in range(i+1, len(available_conds)):
            c1 = available_conds[i]
            c2 = available_conds[j]

            d1, e1 = km_results[c1]["durations"], km_results[c1]["events"]
            d2, e2 = km_results[c2]["durations"], km_results[c2]["events"]

            chi2, lr_p = log_rank_test(d1, e1, d2, e2)
            perm_p = permutation_test_median(d1, e1, d2, e2)

            sig = "✅" if lr_p < 0.05 else "❌"
            stats_markdown.append(f"| {c1} vs {c2} | {lr_p:.4e} | {perm_p:.4e} | {sig} |")

    # Markdown Report Generation
    md_path = output_dir / "CLIFF_RIGOR.md"
    with open(md_path, "w") as f:
        f.write("# Grokking Cliff Rigor Analysis\n\n")
        f.write("This report re-evaluates the headline grokking cliff results with rigorous statistical bounds, changepoint detection, and survival analysis to properly handle finite-compute censoring.\n\n")

        f.write("## 1. Grokking Survival Curves\n\n")
        f.write("We frame 'did not grok' as a right-censored event using a Kaplan-Meier estimator. The survival curves below represent the probability of a model *not* having grokked by a given step.\n\n")
        f.write("![Survival Curves](./survival_curves.png)\n\n")

        f.write("## 2. Grokking Step Distributions\n\n")
        f.write("For runs that did grok, the distributions of detected changepoints (grokking steps) are shown below:\n\n")
        f.write("![Step Distributions](./grok_step_distributions.png)\n\n")

        f.write("## 3. Aggregate Statistics\n\n")
        f.write("| Condition | Runs | Grokked | Median Grok Step | IQR | Max Grok Step |\n")
        f.write("|-----------|------|---------|------------------|-----|---------------|\n")

        for cond in available_conds:
            st = agg_results[cond]["stats"]
            runs = st["num_runs"]
            grokked = st["num_grokked"]
            med = f"{st['median']:.1f}" if st['median'] is not None else "N/A"
            iqr = f"{st['iqr']:.1f}" if st['iqr'] is not None else "N/A"
            mx = f"{st['max_grok_step']:.1f}" if st['max_grok_step'] is not None else "N/A"

            f.write(f"| {cond} | {runs} | {grokked} | {med} | {iqr} | {mx} |\n")

        f.write("\n")
        f.write("## 4. Method Comparison\n\n")
        f.write("We compared three changepoint detection methods on the first `pure` data run:\n\n")
        if method_comparisons:
            comp = method_comparisons[0]
            f.write(f"- **Seed**: {comp['seed']}\n")
            f.write(f"- **Piecewise Linear (Min SSE)**: {comp['piecewise']}\n")
            f.write(f"- **Smoothed Derivative**: {comp['smoothed_deriv']}\n")
            f.write(f"- **Bayesian Online (BOCD)**: {comp['bocd']}\n\n")
            f.write("The piecewise linear method was chosen as the robust default for the aggregate statistics above.\n")

        f.write("\n")
        f.write("\n".join(stats_markdown))
        f.write("\n")

    print(f"Analysis complete. Report saved to {md_path}")

if __name__ == "__main__":
    results_dir = Path("results/multi_seed")
    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist. Please run a multi-seed experiment first.")
    else:
        run_analysis(results_dir, Path("analysis"))
