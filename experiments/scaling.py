import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train, TrainConfig
from src.data import DatasetConfig
from src.phase_detection import detect_grokking_phase, detect_collapse_onset, compute_critical_points

@dataclass
class ScalingExperimentConfig:
    d_models: List[int] = None
    primes: List[int] = None
    collapse_levels: List[float] = None
    max_steps: int = 10000
    eval_every: int = 500
    output_dir: str = "results/scaling"
    seed: int = 42

    def __post_init__(self):
        if self.d_models is None:
            self.d_models = [32, 64, 128, 256, 512]
        if self.primes is None:
            self.primes = [29, 59, 97, 113, 127]
        if self.collapse_levels is None:
            # 10 granular levels
            self.collapse_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def run_scaling_experiments(config: ScalingExperimentConfig):
    """
    Run nested loops over d_model, primes, and collapse levels.
    Saves results to config.output_dir.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    results = {}

    total_runs = len(config.d_models) * len(config.primes) * len(config.collapse_levels)
    current_run = 0

    for p in config.primes:
        results[p] = {}
        for d in config.d_models:
            results[p][d] = {}
            for c_level in config.collapse_levels:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running prime={p}, d_model={d}, collapse_level={c_level}")

                condition_name = f"p{p}_d{d}_c{c_level:.2f}"

                train_config = TrainConfig(
                    prime=p,
                    d_model=d,
                    collapse_level=c_level,
                    collapse_severity=0.5, # Fixed severity
                    condition_name=condition_name,
                    output_dir=os.path.join(config.output_dir, "runs"),
                    max_steps=config.max_steps,
                    eval_every=config.eval_every,
                    seed=config.seed
                )

                # Try to run training
                try:
                    state = train(train_config)
                    test_accs = [entry["test_acc"] for entry in state.history]

                    # Detect grokking point
                    grok_step_idx = detect_grokking_phase(test_accs, threshold=0.95, window_size=min(5, len(test_accs)//2))

                    grok_step = None
                    if grok_step_idx is not None:
                        grok_step = state.history[grok_step_idx]["step"]

                    # Calculate baseline onset against pure run (c_level=0)
                    baseline_onset = None
                    if c_level > 0 and 0.0 in results[p][d] and "test_accs" in results[p][d][0.0]:
                        baseline_accs = results[p][d][0.0]["test_accs"]
                        onset_idx = detect_collapse_onset(test_accs, baseline_accs)
                        if onset_idx is not None:
                            baseline_onset = state.history[onset_idx]["step"]

                    results[p][d][c_level] = {
                        "grokked": state.grokked,
                        "grok_step": grok_step,
                        "collapse_onset_step": baseline_onset,
                        "final_test_acc": state.test_acc,
                        "test_accs": test_accs  # Keep for critical point analysis, filter before saving
                    }

                except Exception as e:
                    print(f"Error in run: {e}")
                    results[p][d][c_level] = {
                        "error": str(e)
                    }

    # Save results
    # Run critical point analysis on the results
    critical_points = {}
    for p, d_dict in results.items():
        critical_points[p] = {}
        for d, c_dict in d_dict.items():
            # Build severity -> acc curve map
            test_acc_dict = {}
            for c_level, run_data in c_dict.items():
                if "test_accs" in run_data:
                    test_acc_dict[c_level] = run_data["test_accs"]

            if test_acc_dict:
                crit_pts = compute_critical_points(test_acc_dict, grokking_threshold=0.95)

                # We need to make sure we don't try to JSON serialize 'None' keys or strange types later.
                # Just store serializable info.
                safe_pts = {
                    "collapse_threshold": crit_pts["collapse_threshold"],
                    "recovery_potential": crit_pts["recovery_potential"],
                }
                critical_points[p][d] = safe_pts

    # Save results
    with open(os.path.join(config.output_dir, "scaling_results.json"), "w") as f:
        # Don't save the full history in the main JSON to save space, just the summary
        summary_results = {}
        for p, d_dict in results.items():
            summary_results[p] = {}
            for d, c_dict in d_dict.items():
                summary_results[p][d] = {}
                for c, res in c_dict.items():
                    summary_results[p][d][c] = {k: v for k, v in res.items() if k != "test_accs" and k != "test_accs_full"}

        json.dump(summary_results, f, indent=2)

    with open(os.path.join(config.output_dir, "critical_points.json"), "w") as f:
        json.dump(critical_points, f, indent=2)

    return results

def plot_scaling_laws(results: Dict, output_dir: str):
    """
    Generate scaling law plots from experiment results.
    Plots grokking step vs model size for each collapse level.
    """
    os.makedirs(output_dir, exist_ok=True)

    for p_str, d_dict in results.items():
        plt.figure(figsize=(10, 6))

        # Get all unique collapse levels
        all_c_levels = set()
        for d_str, c_dict in d_dict.items():
            for c_str in c_dict.keys():
                all_c_levels.add(float(c_str))

        c_levels = sorted(list(all_c_levels))

        # We need to plot d_model vs grok_step for each collapse level
        for c_level in c_levels:
            d_models = []
            grok_steps = []

            for d_str, c_dict in d_dict.items():
                d = int(d_str)
                c_level_str = str(c_level) # JSON keys are strings

                # Handle potential float formatting differences in JSON keys
                if c_level_str not in c_dict:
                    # Find closest match
                    closest_k = None
                    min_diff = float('inf')
                    for k in c_dict.keys():
                        try:
                            diff = abs(float(k) - c_level)
                            if diff < min_diff:
                                min_diff = diff
                                closest_k = k
                        except:
                            pass
                    if min_diff < 1e-5:
                        c_level_str = closest_k

                if c_level_str in c_dict:
                    res = c_dict[c_level_str]
                    if not res.get("error") and res.get("grokked") and res.get("grok_step") is not None:
                        d_models.append(d)
                        grok_steps.append(res["grok_step"])

            if d_models:
                # Sort by d_model
                sorted_indices = np.argsort(d_models)
                d_models = np.array(d_models)[sorted_indices]
                grok_steps = np.array(grok_steps)[sorted_indices]

                plt.plot(d_models, grok_steps, marker='o', label=f'Collapse: {c_level:.2f}')

        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.xlabel('Model Size (d_model)')
        plt.ylabel('Grokking Step')
        plt.title(f'Grokking Step vs Model Size (Prime={p_str})')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        plt.savefig(os.path.join(output_dir, f'scaling_law_p{p_str}.png'))
        plt.close()

if __name__ == "__main__":
    # Test execution
    config = ScalingExperimentConfig(
        d_models=[32, 64],
        primes=[29],
        collapse_levels=[0.0, 0.1],
        max_steps=1000,
        eval_every=100
    )
    # run_scaling_experiments(config)
