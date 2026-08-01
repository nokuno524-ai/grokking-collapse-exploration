import json
import numpy as np
import scipy.stats as st
from pathlib import Path

def aggregate_scaling_results(scaling_results_dir: str):
    """
    Parses scaling results and identifies:
    - Confidence intervals for grokking transitions
    - Collapse point of no return
    - Statistical comparison (grokking vs non-grokking)
    """
    root = Path(scaling_results_dir)
    if not root.exists():
        print(f"Directory {scaling_results_dir} does not exist.")
        return

    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    stats = {}

    # 1. Grokking Transition CIs
    for cond in conditions:
        cond_dir = root / cond
        if not cond_dir.exists():
            continue

        grokking_steps = []
        final_accs = []

        # Iterate over seeds
        for seed_dir in cond_dir.glob("seed_*"):
            results_file = seed_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)

                final_accs.append(data.get("final_test_acc", 0.0))
                if data.get("grokked", False):
                    grokking_steps.append(data.get("grokking_step", 0))

        stats[cond] = {
            "grokking_rate": len(grokking_steps) / len(final_accs) if final_accs else 0,
            "mean_grok_step": np.mean(grokking_steps) if grokking_steps else None,
            "grok_step_ci": st.t.interval(0.95, len(grokking_steps)-1, loc=np.mean(grokking_steps), scale=st.sem(grokking_steps)) if len(grokking_steps) > 1 else None,
            "mean_final_acc": np.mean(final_accs) if final_accs else 0,
        }

    # 2. Point of No Return
    # A condition where grokking_rate is 0 is considered past the point of no return.
    point_of_no_return = None
    for cond in conditions:
        if cond in stats and stats[cond]["grokking_rate"] == 0.0:
            point_of_no_return = cond
            break

    print("\n--- Scaling Analysis Report ---")
    print("1. Grokking Transition & CIs:")
    for cond, stat in stats.items():
        if stat['mean_grok_step']:
            ci_str = f"[{stat['grok_step_ci'][0]:.1f}, {stat['grok_step_ci'][1]:.1f}]" if stat['grok_step_ci'] else "N/A"
            print(f"  {cond:16s}: Grok Rate {stat['grokking_rate']*100:3.0f}% | Mean Step {stat['mean_grok_step']:.1f} 95% CI {ci_str}")
        else:
            print(f"  {cond:16s}: Grok Rate {stat['grokking_rate']*100:3.0f}% | NO GROKKING")

    print("\n2. Collapse Point of No Return:")
    print(f"  The model permanently loses the ability to grok starting at condition: {point_of_no_return}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/scaling")
    args = parser.parse_args()

    aggregate_scaling_results(args.results_dir)
