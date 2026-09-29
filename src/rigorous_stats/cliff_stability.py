import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    from .stats_module import bootstrap_ci, permutation_test, holm_correction
except ImportError:
    from src.rigorous_stats.stats_module import bootstrap_ci, permutation_test, holm_correction

def find_cliff_location(history_data: List[Dict[str, Any]], threshold: float = 0.95) -> Optional[int]:
    """
    Find the exact step where test accuracy crosses the grokking threshold for the first time.

    Args:
        history_data: List of dictionaries containing training history (e.g. from results.json).
        threshold: The test accuracy threshold indicating grokking.

    Returns:
        The step number where grokking occurred, or None if it never crossed the threshold.
    """
    for entry in history_data:
        if entry.get("test_acc", 0.0) >= threshold:
            return entry.get("step")
    return None

def calculate_steepness(history_data: List[Dict[str, Any]], cliff_step: int, window: int = 5) -> Optional[float]:
    """
    Calculate the steepness of the cliff around the grokking step.
    Computes the test accuracy change over the steps before and after the cliff.

    Args:
        history_data: List of dictionaries containing training history.
        cliff_step: The step where the cliff occurred.
        window: The number of historical log entries to look backwards/forwards.

    Returns:
        Steepness as accuracy change per step, or None if data is insufficient.
    """
    if not history_data:
        return None

    # Find the index of the cliff step
    idx = -1
    for i, entry in enumerate(history_data):
        if entry.get("step") == cliff_step:
            idx = i
            break

    if idx == -1:
        return None

    start_idx = max(0, idx - window)
    end_idx = min(len(history_data) - 1, idx + window)

    start_entry = history_data[start_idx]
    end_entry = history_data[end_idx]

    start_step = start_entry.get("step")
    end_step = end_entry.get("step")
    start_acc = start_entry.get("test_acc", 0.0)
    end_acc = end_entry.get("test_acc", 0.0)

    step_diff = end_step - start_step
    if step_diff <= 0:
        return None

    return (end_acc - start_acc) / step_diff

def plot_cliff_positions(stability_results: Dict[str, Dict[str, Any]], output_path: str):
    """
    Generate a box plot visualizing the distribution of per-seed cliff locations for each condition.

    Args:
        stability_results: Dictionary of condition names to their analyzed stability results.
        output_path: Path to save the plot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
        return

    conditions = []
    cliff_data = []

    known_order = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    ordered_conds = sorted(stability_results.keys(), key=lambda x: known_order.index(x) if x in known_order else 999)

    for cond in ordered_conds:
        res = stability_results[cond]
        if res["num_grokked"] > 0:
            conditions.append(cond)
            cliff_data.append(res["cliff_locations"])

    if not conditions:
        print("No conditions with grokked runs found. Skipping visualization.")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(cliff_data, labels=conditions, patch_artist=True)
    plt.title('Grokking Cliff Locations by Condition')
    plt.ylabel('Step')
    plt.xlabel('Condition')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Cliff positions visualization saved to {output_path}")

def analyze_cliff_stability(runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the stability of cliff locations and steepness across multiple runs (seeds).

    Args:
        runs_data: List of results dictionaries for multiple seeds of the same condition.

    Returns:
        Dictionary containing summary statistics.
    """
    cliff_locations = []
    steepness_values = []

    for run in runs_data:
        # Prefer pre-computed grokking step if available, else recalculate
        grokking_step = run.get("grokking_step")
        if grokking_step is None and "history" in run:
            grokking_step = find_cliff_location(run["history"])

        if grokking_step is not None:
            cliff_locations.append(grokking_step)

            if "history" in run:
                steepness = calculate_steepness(run["history"], grokking_step)
                if steepness is not None:
                    steepness_values.append(steepness)

    num_runs = len(runs_data)
    num_grokked = len(cliff_locations)

    result = {
        "num_runs": num_runs,
        "num_grokked": num_grokked,
        "grokking_prob": num_grokked / num_runs if num_runs > 0 else 0.0,
        "cliff_locations": cliff_locations,
        "steepness_values": steepness_values,
    }

    if num_grokked > 0:
        result["mean_cliff"] = np.mean(cliff_locations)
        result["std_cliff"] = np.std(cliff_locations)

        if steepness_values:
            result["mean_steepness"] = np.mean(steepness_values)
            result["std_steepness"] = np.std(steepness_values)

        if num_grokked > 1:
            try:
                ci_lower, ci_upper = bootstrap_ci(cliff_locations, num_bootstraps=1000)
                result["ci_lower"] = ci_lower
                result["ci_upper"] = ci_upper

                if len(steepness_values) > 1:
                    steep_lower, steep_upper = bootstrap_ci(steepness_values, num_bootstraps=1000)
                    result["steepness_ci_lower"] = steep_lower
                    result["steepness_ci_upper"] = steep_upper
            except Exception:
                pass

    return result

def aggregate_and_report_results(results_dir: str, output_markdown: str = "analysis/rigorous_summary.md"):
    """
    Aggregate multi-seed results, compute rigorous statistics, and generate a markdown report.
    Expects directory structure: results_dir/<seed>/<condition>/results.json
    """
    root_dir = Path(results_dir)
    if not root_dir.exists():
        print(f"Results directory {results_dir} does not exist.")
        return

    # Gather data
    conditions_data = defaultdict(list)

    for seed_dir in root_dir.iterdir():
        if not seed_dir.is_dir():
            continue

        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue

            results_file = cond_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, "r") as f:
                        data = json.load(f)
                    conditions_data[cond_dir.name].append(data)
                except Exception as e:
                    print(f"Error loading {results_file}: {e}")

    if not conditions_data:
        print("No results found to aggregate.")
        return

    # Analyze stability per condition
    stability_results = {}
    for cond_name, runs in conditions_data.items():
        stability_results[cond_name] = analyze_cliff_stability(runs)

    # Generate Plot
    plot_output_path = out_path = Path(output_markdown).parent / "cliff_positions.png"
    plot_cliff_positions(stability_results, str(plot_output_path))

    # Write Markdown Report
    out_path = Path(output_markdown)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# Rigorous Grokking Analysis\n\n")
        f.write("## Cliff Stability (Grokking Step across seeds)\n\n")

        f.write("| Condition | Runs | Grokked | Mean Step | Std Dev | 95% CI | Mean Steepness | Steepness 95% CI |\n")
        f.write("|-----------|------|---------|-----------|---------|--------|----------------|------------------|\n")

        # Sort conditions by name heuristically if SEVERITY_ORDER matches
        ordered_conds = sorted(stability_results.keys())
        # Try to sort by known order if possible
        known_order = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
        ordered_conds = sorted(ordered_conds, key=lambda x: known_order.index(x) if x in known_order else 999)

        for cond in ordered_conds:
            res = stability_results[cond]
            runs = res["num_runs"]
            grokked = res["num_grokked"]

            if grokked > 0:
                mean_val = f"{res['mean_cliff']:.1f}"
                std_val = f"{res.get('std_cliff', 0.0):.1f}"
                ci_str = f"[{res.get('ci_lower', 0.0):.1f}, {res.get('ci_upper', 0.0):.1f}]" if "ci_lower" in res else "N/A"

                steep_mean = f"{res.get('mean_steepness', 0.0):.6f}" if "mean_steepness" in res else "N/A"
                steep_ci = f"[{res.get('steepness_ci_lower', 0.0):.6f}, {res.get('steepness_ci_upper', 0.0):.6f}]" if "steepness_ci_lower" in res else "N/A"
            else:
                mean_val = "N/A"
                std_val = "N/A"
                ci_str = "N/A"
                steep_mean = "N/A"
                steep_ci = "N/A"

            f.write(f"| {cond} | {runs} | {grokked} | {mean_val} | {std_val} | {ci_str} | {steep_mean} | {steep_ci} |\n")

        # Example Permutation Test between "pure" and others if possible
        if "pure" in stability_results and stability_results["pure"]["num_grokked"] > 1:
            pure_cliffs = stability_results["pure"]["cliff_locations"]

            f.write("\n## Permutation Tests vs Pure Condition\n\n")
            f.write("| Condition | Permutation Test p-value | Holm Corrected p-value | Significant (p<0.05)? |\n")
            f.write("|-----------|--------------------------|------------------------|-----------------------|\n")

            comparisons = []
            raw_p_values = []

            for cond in ordered_conds:
                if cond == "pure":
                    continue
                if stability_results[cond]["num_grokked"] > 1:
                    cond_cliffs = stability_results[cond]["cliff_locations"]
                    try:
                        p_val = permutation_test(pure_cliffs, cond_cliffs)
                        comparisons.append((cond, p_val))
                        raw_p_values.append(p_val)
                    except Exception:
                        pass

            if raw_p_values:
                corrected_p_values = holm_correction(raw_p_values)
                for (cond, raw_p), corr_p in zip(comparisons, corrected_p_values):
                    sig = "Yes" if corr_p < 0.05 else "No"
                    f.write(f"| {cond} | {raw_p:.4f} | {corr_p:.4f} | {sig} |\n")
            else:
                f.write("| N/A | N/A | N/A | N/A |\n")

    print(f"Rigorous report generated at {output_markdown}")
