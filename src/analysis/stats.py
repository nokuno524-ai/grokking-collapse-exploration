"""
Statistical analysis tools for grokking cliff trajectories.
Provides changepoint detection, bootstrap confidence intervals, and hypothesis testing.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression


def detect_grokking_cliff(test_acc_array: np.ndarray) -> Optional[Tuple[int, float]]:
    """
    Detect the index where the grokking cliff occurs using a heuristic approximation
    of a piecewise linear fit (or CUSUM). We find the point of maximum increase in
    a monotonic fit to the data, which works well for step-like grokking transitions.

    Args:
        test_acc_array: 1D numpy array of test accuracies over time.

    Returns:
        A tuple of (cliff_index, transition_magnitude) or None if no transition occurred.
    """
    if len(test_acc_array) < 5:
        return None

    initial_acc = np.mean(test_acc_array[:max(1, len(test_acc_array)//10)])
    final_acc = np.mean(test_acc_array[-max(1, len(test_acc_array)//10):])

    # If the total jump is very small, it didn't grok (e.g. diff < 0.4)
    if final_acc - initial_acc < 0.4:
        return None

    # Use isotonic regression to smooth the trajectory into a monotonic step function
    iso_reg = IsotonicRegression(y_min=initial_acc, y_max=final_acc, increasing=True)
    smoothed = iso_reg.fit_transform(np.arange(len(test_acc_array)), test_acc_array)

    # Find the maximum derivative in the smoothed data (the cliff)
    diffs = np.diff(smoothed)
    cliff_idx = int(np.argmax(diffs))
    magnitude = float(diffs[cliff_idx])

    # To be conservative on the exact step of the jump, we check if the jump is substantial
    if diffs[cliff_idx] < 0.05:  # small gradual slopes might not be a cliff
        # fallback to threshold crossing
        target = initial_acc + 0.5 * (final_acc - initial_acc)
        crosses = smoothed > target
        if np.any(crosses):
            idx = int(np.argmax(crosses))
            # Calculate a generic magnitude for the fallback
            mag = float(smoothed[min(len(smoothed)-1, idx+1)] - smoothed[max(0, idx-1)])
            return idx, mag
        return None

    return cliff_idx, magnitude


def bootstrap_cliff(trajectories: List[np.ndarray], n_bootstraps: int = 1000, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute bootstrap confidence intervals for the transition step and magnitude.

    Args:
        trajectories: List of 1D numpy arrays containing test accuracies.
        n_bootstraps: Number of bootstrap resamples.
        alpha: Significance level for the confidence interval.

    Returns:
        Dictionary containing mean cliff index/magnitude, bounds, and fraction of runs that grokked.
    """
    cliff_indices = []
    cliff_magnitudes = []

    for traj in trajectories:
        res = detect_grokking_cliff(traj)
        if res is not None:
            idx, mag = res
            cliff_indices.append(idx)
            cliff_magnitudes.append(mag)

    n_grokked = len(cliff_indices)
    n_total = len(trajectories)
    grok_rate = n_grokked / n_total if n_total > 0 else 0.0

    result = {
        "grok_rate": grok_rate,
        "n_grokked": n_grokked,
        "n_total": n_total,
        "mean_step": None,
        "ci_step_lower": None,
        "ci_step_upper": None,
        "mean_magnitude": None,
        "ci_mag_lower": None,
        "ci_mag_upper": None,
    }

    if n_grokked < 2:
        if n_grokked == 1:
            result["mean_step"] = float(cliff_indices[0])
            result["mean_magnitude"] = float(cliff_magnitudes[0])
        return result

    cliff_indices = np.array(cliff_indices)
    cliff_magnitudes = np.array(cliff_magnitudes)
    bootstrapped_steps = []
    bootstrapped_mags = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_grokked, size=n_grokked, replace=True)
        bootstrapped_steps.append(np.mean(cliff_indices[indices]))
        bootstrapped_mags.append(np.mean(cliff_magnitudes[indices]))

    result["mean_step"] = float(np.mean(cliff_indices))
    result["ci_step_lower"] = float(np.percentile(bootstrapped_steps, 100 * (alpha / 2)))
    result["ci_step_upper"] = float(np.percentile(bootstrapped_steps, 100 * (1 - alpha / 2)))

    result["mean_magnitude"] = float(np.mean(cliff_magnitudes))
    result["ci_mag_lower"] = float(np.percentile(bootstrapped_mags, 100 * (alpha / 2)))
    result["ci_mag_upper"] = float(np.percentile(bootstrapped_mags, 100 * (1 - alpha / 2)))

    return result

def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def compare_conditions(group_a: List[np.ndarray], group_b: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compare two experimental conditions statistically.

    Args:
        group_a: List of test accuracy trajectories for condition A.
        group_b: List of test accuracy trajectories for condition B.

    Returns:
        Dictionary of statistical comparisons (Mann-Whitney U p-value, effect sizes, Wilson CIs).
    """
    results_a = [detect_grokking_cliff(t) for t in group_a]
    results_b = [detect_grokking_cliff(t) for t in group_b]

    valid_a = [r[0] for r in results_a if r is not None]
    valid_b = [r[0] for r in results_b if r is not None]

    final_accs_a = [t[-1] for t in group_a if len(t) > 0]
    final_accs_b = [t[-1] for t in group_b if len(t) > 0]

    # Wilson score intervals for grokking proportions
    n_a = len(results_a)
    n_b = len(results_b)
    succ_a = len(valid_a)
    succ_b = len(valid_b)

    ci_a = wilson_score_interval(succ_a, n_a) if n_a > 0 else (0.0, 0.0)
    ci_b = wilson_score_interval(succ_b, n_b) if n_b > 0 else (0.0, 0.0)

    # Proportion difference test (z-test approx)
    prop_p_value = None
    if n_a > 0 and n_b > 0:
        p_a = succ_a / n_a
        p_b = succ_b / n_b
        p_pool = (succ_a + succ_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        if se > 0:
            z = (p_a - p_b) / se
            prop_p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        elif p_a != p_b:
            prop_p_value = 0.0
        else:
            prop_p_value = 1.0

    # Mann-Whitney U test for step distribution differences & rank biserial correlation
    mw_p_value = None
    step_effect_size = None
    if len(valid_a) >= 3 and len(valid_b) >= 3:
        try:
            u_stat, mw_p_value = stats.mannwhitneyu(valid_a, valid_b, alternative='two-sided')
            if np.isnan(mw_p_value):
                mw_p_value = 1.0
            n1 = len(valid_a)
            n2 = len(valid_b)
            # Rank biserial correlation
            step_effect_size = 1 - (2 * u_stat) / (n1 * n2)
        except ValueError:
            mw_p_value = 1.0
            step_effect_size = 0.0

    # Final accuracy distribution comparisons
    final_acc_mw_p = None
    if len(final_accs_a) >= 3 and len(final_accs_b) >= 3:
         try:
            _, final_acc_mw_p = stats.mannwhitneyu(final_accs_a, final_accs_b, alternative='two-sided')
            if np.isnan(final_acc_mw_p):
                final_acc_mw_p = 1.0
         except ValueError:
            final_acc_mw_p = 1.0

    return {
        "grok_rate_a": succ_a / n_a if n_a > 0 else 0.0,
        "grok_rate_b": succ_b / n_b if n_b > 0 else 0.0,
        "wilson_ci_a": ci_a,
        "wilson_ci_b": ci_b,
        "prop_p_value": prop_p_value,
        "mw_p_value": mw_p_value,
        "step_effect_size": step_effect_size,
        "mean_step_a": np.mean(valid_a) if valid_a else None,
        "mean_step_b": np.mean(valid_b) if valid_b else None,
        "final_acc_mean_a": np.mean(final_accs_a) if final_accs_a else None,
        "final_acc_mean_b": np.mean(final_accs_b) if final_accs_b else None,
        "final_acc_mw_p": final_acc_mw_p
    }


def generate_summary_markdown(results_dir: str, output_file: str) -> None:
    """
    Load results from multi-seed runner, run statistical tests, and generate a markdown summary.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist.")
        return

    # Discover conditions and group by condition -> list of test accuracies
    conditions_data = {}

    for seed_dir in results_path.iterdir():
        if not seed_dir.is_dir():
            continue
        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            cond_name = cond_dir.name
            res_file = cond_dir / "results.json"
            if res_file.exists():
                try:
                    with open(res_file, 'r') as f:
                        data = json.load(f)
                    history = data.get("history", [])
                    test_accs = [h.get("test_acc", 0.0) for h in history]
                    if test_accs:
                        conditions_data.setdefault(cond_name, []).append(np.array(test_accs))
                except Exception as e:
                    print(f"Failed to read {res_file}: {e}")

    if not conditions_data:
        print("No valid results found.")
        return

    # Calculate individual bootstrap stats
    stats_summary = {}
    for cond_name, trajectories in conditions_data.items():
        stats_summary[cond_name] = bootstrap_cliff(trajectories)

    # Generate Markdown
    md = [
        "# Statistical Summary of Grokking Cliffs",
        "",
        "## Overall Grokking Rates, Transition Steps, and Magnitudes",
        "",
        "| Condition | N | Grok Rate (Wilson 95% CI) | Mean Transition Step (95% CI) | Mean Transition Magnitude (95% CI) |",
        "|---|---|---|---|---|"
    ]

    for cond_name in sorted(conditions_data.keys()):
        s = stats_summary[cond_name]
        ci_lower, ci_upper = wilson_score_interval(s['n_grokked'], s['n_total'])
        rate_str = f"{s['grok_rate']:.2f} ({ci_lower:.2f} - {ci_upper:.2f})"

        if s['mean_step'] is not None and s['ci_step_lower'] is not None:
            step_str = f"{s['mean_step']:.1f} ({s['ci_step_lower']:.1f} - {s['ci_step_upper']:.1f})"
        elif s['mean_step'] is not None:
            step_str = f"{s['mean_step']:.1f} (N/A)"
        else:
            step_str = "N/A"

        if s['mean_magnitude'] is not None and s['ci_mag_lower'] is not None:
            mag_str = f"{s['mean_magnitude']:.3f} ({s['ci_mag_lower']:.3f} - {s['ci_mag_upper']:.3f})"
        elif s['mean_magnitude'] is not None:
            mag_str = f"{s['mean_magnitude']:.3f} (N/A)"
        else:
            mag_str = "N/A"

        md.append(f"| {cond_name} | {s['n_total']} | {rate_str} | {step_str} | {mag_str} |")

    md.extend(["", "## Statistical Comparisons (vs pure)", ""])

    if "pure" in conditions_data:
        pure_data = conditions_data["pure"]
        md.append("| Condition | Prop p-value | Transition Step MW p-value (Effect Size) | Final Acc MW p-value | Significant? |")
        md.append("|---|---|---|---|---|")

        for cond_name, cond_data in conditions_data.items():
            if cond_name == "pure":
                continue
            comp = compare_conditions(pure_data, cond_data)
            prop_p = f"{comp['prop_p_value']:.4f}" if comp['prop_p_value'] is not None else "N/A"

            if comp['mw_p_value'] is not None and comp['step_effect_size'] is not None:
                 mw_p = f"{comp['mw_p_value']:.4f} ({comp['step_effect_size']:.2f})"
            else:
                 mw_p = "N/A"

            final_acc_p = f"{comp['final_acc_mw_p']:.4f}" if comp['final_acc_mw_p'] is not None else "N/A"

            sig = "Yes" if (comp['prop_p_value'] is not None and comp['prop_p_value'] < 0.05) or \
                           (comp['mw_p_value'] is not None and comp['mw_p_value'] < 0.05) or \
                           (comp['final_acc_mw_p'] is not None and comp['final_acc_mw_p'] < 0.05) else "No"

            md.append(f"| {cond_name} | {prop_p} | {mw_p} | {final_acc_p} | {sig} |")
    else:
        md.append("No 'pure' condition found for baseline comparison.")

    md.extend([
        "",
        "## Key Claims and Confidence",
        "",
        "**Supported Claims:**",
        "- (Example) 'Severe collapse models completely fail to grok.' (Requires grok rate ~ 0 with tight CI).",
        "",
        "**Claims Not Supported by Current Seed Count:**",
        "If N is small (e.g. N < 5) and confidence intervals overlap significantly, we cannot confidently assert the difference in transition step distributions."
    ])

    with open(output_file, 'w') as f:
        f.write('\n'.join(md))
    print(f"Summary generated at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a statistical summary from multi-seed results.")
    parser.add_argument("results_dir", type=str, help="Path to the multi-seed results directory.")
    parser.add_argument("output_file", type=str, nargs='?', default="stats_summary.md", help="Path to the output Markdown file.")
    args = parser.parse_args()

    generate_summary_markdown(args.results_dir, args.output_file)
