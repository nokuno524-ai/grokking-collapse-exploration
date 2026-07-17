import os
import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import argparse

def analyze_scaling(results_dir: str):
    """Analyze scaling laws for grokking delay vs model size/collapse level."""
    summary_path = Path(results_dir) / "scaling" / "scaling_summary.json"
    if not summary_path.exists():
        print("Scaling summary not found.")
        return

    with open(summary_path, 'r') as f:
        data = json.load(f)

    print(f"\\n{'='*40}\\nScaling Analysis\\n{'='*40}")
    # Very basic Chinchilla-style fit on grokking step vs params
    # Step = A * Width^alpha * Depth^beta + B * Collapse
    # We will just print basic correlations for simplicity

    # Filter for models that actually grokked
    grokked_data = [d for d in data if d['grokked'] and d['grokking_step'] is not None]
    if len(grokked_data) < 3:
        print("Not enough successful grokking runs to fit scaling laws.")
        return

    widths = np.array([d['width'] for d in grokked_data])
    depths = np.array([d['depth'] for d in grokked_data])
    collapses = np.array([d['collapse_level'] for d in grokked_data])
    steps = np.array([d['grokking_step'] for d in grokked_data])

    # Simple linear regression to find correlation
    from sklearn.linear_model import LinearRegression
    X = np.stack([np.log(widths), np.log(depths), collapses], axis=-1)
    y = np.log(steps)

    reg = LinearRegression().fit(X, y)
    print(f"Scaling Law fit (log Step ~ alpha*logW + beta*logD + gamma*C)")
    print(f"alpha (Width effect): {reg.coef_[0]:.4f}")
    print(f"beta (Depth effect): {reg.coef_[1]:.4f}")
    print(f"gamma (Collapse effect): {reg.coef_[2]:.4f}")
    print(f"R^2 score: {reg.score(X, y):.4f}")


def analyze_curriculum(results_dir: str):
    """Analyze the effect of curriculum learning schedules."""
    summary_path = Path(results_dir) / "curriculum" / "curriculum_summary.json"
    if not summary_path.exists():
        print("Curriculum summary not found.")
        return

    with open(summary_path, 'r') as f:
        data = json.load(f)

    print(f"\\n{'='*40}\\nCurriculum Analysis\\n{'='*40}")

    # Group by schedule
    schedules = {}
    for d in data:
        sched = d['schedule']
        if sched not in schedules:
            schedules[sched] = []
        if d['grokked']:
            schedules[sched].append(d['grokking_step'])

    for sched, steps in schedules.items():
        if steps:
            mean_step = np.mean(steps)
            std_step = np.std(steps)
            success_rate = len(steps) / sum(1 for d in data if d['schedule'] == sched)
            print(f"Schedule: {sched:10s} | Success: {success_rate*100:5.1f}% | Avg Step: {mean_step:8.1f} ± {std_step:6.1f}")
        else:
            print(f"Schedule: {sched:10s} | Success:   0.0%")


def analyze_threshold(results_dir: str):
    """Analyze phase transition threshold with bootstrap CI."""
    summary_path = Path(results_dir) / "threshold" / "thresholds_summary.json"
    if not summary_path.exists():
        print("Threshold summary not found.")
        return

    with open(summary_path, 'r') as f:
        data = json.load(f)

    print(f"\\n{'='*40}\\nThreshold Analysis\\n{'='*40}")

    thresholds = [d['threshold'] for d in data['thresholds']]

    if not thresholds:
        print("No thresholds found.")
        return

    mean_t = np.mean(thresholds)

    # Bootstrap CI
    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(thresholds, size=len(thresholds), replace=True)
        boot_means.append(np.mean(sample))

    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    print(f"Mean Collapse Threshold: {mean_t:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Data points (seeds): {len(thresholds)}")


def run_all_analysis(results_dir: str):
    analyze_scaling(results_dir)
    analyze_curriculum(results_dir)
    analyze_threshold(results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    run_all_analysis(args.results_dir)
