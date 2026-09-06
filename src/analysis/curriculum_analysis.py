"""
Analysis tools for curriculum rescue experiments.
Parses results from src/run_curriculum.py and generates a markdown report.
Handles right-censored runs (runs that hit max_steps without grokking).
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def load_curriculum_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all results.json files from the curriculum run directories."""
    base_dir = Path(results_dir)
    results = []
    if not base_dir.exists():
        return results

    for p in base_dir.rglob("results.json"):
        with open(p, "r") as f:
            data = json.load(f)
            # Add directory name to track run configuration
            data["run_name"] = p.parent.name
            results.append(data)
    return results


def summarize_results(results: List[Dict[str, Any]]) -> str:
    """Generate a Markdown summary table of the curriculum results."""
    if not results:
        return "No results found."

    # Group by schedule and switch step
    summary = {}
    for r in results:
        cfg = r["config"]
        sched = cfg.get("schedule_type", "unknown")
        switch = cfg.get("switch_step", 0)
        key = f"{sched}_{switch}"

        if key not in summary:
            summary[key] = {
                "schedule": sched,
                "switch_step": switch,
                "n_runs": 0,
                "n_grokked": 0,
                "grok_steps": [],
                "final_accs": []
            }

        summary[key]["n_runs"] += 1
        if r.get("grokked", False):
            summary[key]["n_grokked"] += 1
            summary[key]["grok_steps"].append(r.get("grokking_step"))
        summary[key]["final_accs"].append(r.get("final_test_acc", 0.0))

    # Build Markdown table
    md = [
        "# Curriculum Rescue Experiments Summary\n",
        "| Schedule | Switch Step | Grokked | Avg Grok Step (if grokked) | Avg Final Acc |",
        "|---|---|---|---|---|"
    ]

    # Sort keys for consistent table ordering
    for key in sorted(summary.keys(), key=lambda k: summary[k]["switch_step"]):
        group = summary[key]
        n = group["n_runs"]
        g = group["n_grokked"]

        g_ratio = f"{g}/{n}"

        if g > 0:
            avg_grok = f"{np.mean(group['grok_steps']):.0f} ± {np.std(group['grok_steps']):.0f}"
        else:
            avg_grok = "N/A (Censored)"

        avg_acc = f"{np.mean(group['final_accs']):.3f}"

        row = f"| {group['schedule']} | {group['switch_step']} | {g_ratio} | {avg_grok} | {avg_acc} |"
        md.append(row)

    return "\n".join(md)


def generate_report(results_dir: str = "results/curriculum_rescue", output_file: str = "analysis/curriculum_report.md"):
    """Main entry point to parse results and generate the report."""
    results = load_curriculum_results(results_dir)
    report_content = summarize_results(results)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(report_content)

    print(f"Report generated at {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/curriculum_rescue")
    parser.add_argument("--output", type=str, default="analysis/curriculum_report.md")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output)
