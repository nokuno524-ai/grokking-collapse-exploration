import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .detectors import threshold_detector, logistic_detector, bootstrap_ci
from .stats import wilson_ci, bootstrap_effect_size

def parse_run(history: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a single run's history.
    Handle NaNs by filtering them out.
    Returns (steps, train_accs, test_accs)
    """
    steps = []
    train_accs = []
    test_accs = []

    for entry in history:
        s = entry.get("step")
        tr = entry.get("train_acc")
        te = entry.get("test_acc")

        # Only keep valid step points without NaNs
        if s is not None and tr is not None and te is not None:
            if not np.isnan(tr) and not np.isnan(te):
                steps.append(s)
                train_accs.append(tr)
                test_accs.append(te)

    return np.array(steps), np.array(train_accs), np.array(test_accs)

def extract_grokking_steps(runs: List[List[Dict]], method="logistic") -> List[Optional[float]]:
    """
    Extract grokking steps for multiple runs using specified method.
    """
    grok_steps = []
    for hist in runs:
        steps, _, test_accs = parse_run(hist)
        if len(steps) == 0:
            grok_steps.append(None)
            continue

        if method == "threshold":
            val = threshold_detector(steps, test_accs)
        elif method == "logistic":
            val = logistic_detector(steps, test_accs)
        else:
            val = threshold_detector(steps, test_accs) # default fallback

        grok_steps.append(val)

    return grok_steps

def generate_multi_seed_report(results_dir: Path, method="logistic") -> str:
    """
    Generate a markdown report from multi-seed runs.
    Expected directory structure: results_dir / <seed> / <condition> / results.json
    """
    # Group by condition name: condition -> list of histories
    condition_runs = defaultdict(list)

    if not results_dir.exists():
        return "Directory not found."

    for seed_dir in results_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue

        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue

            results_path = cond_dir / "results.json"
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    hist = data.get("history", [])
                    condition_runs[cond_dir.name].append(hist)
                except Exception:
                    pass

    # Order conditions
    SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    ordered_conditions = []
    for name in SEVERITY_ORDER:
        if name in condition_runs:
            ordered_conditions.append(name)
    for name in sorted(condition_runs.keys()):
        if name not in ordered_conditions:
            ordered_conditions.append(name)

    # Pre-extract grokking steps for all conditions
    cond_steps = {}
    for cond in ordered_conditions:
        hists = condition_runs[cond]
        if len(hists) > 0:
            cond_steps[cond] = extract_grokking_steps(hists, method=method)

    # Generate report
    lines = [
        "| Condition | Median Step | 95% CI (Step) | n seeds | P(Grok) | 95% CI (P) | Cohen's d (vs pure) |",
        "|-----------|-------------|---------------|---------|---------|------------|---------------------|"
    ]

    pure_steps_valid = []
    if "pure" in cond_steps:
        pure_steps_valid = [s for s in cond_steps["pure"] if s is not None]

    for cond in ordered_conditions:
        if cond not in cond_steps:
            continue

        grok_steps = cond_steps[cond]
        n_seeds = len(grok_steps)

        # Calculate P(Grok)
        successes = sum(1 for s in grok_steps if s is not None)
        p_grok, p_lower, p_upper = wilson_ci(successes, n_seeds)

        p_str = f"{p_grok:.2f}"
        p_ci_str = f"[{p_lower:.2f}, {p_upper:.2f}]"

        # Calculate Step stats on successful runs
        valid_steps = [s for s in grok_steps if s is not None]
        if valid_steps:
            med_step = np.median(valid_steps)

            # Simple bootstrap on the median across seeds
            if len(valid_steps) >= 2:
                boot_meds = []
                for _ in range(1000):
                    b = np.random.choice(valid_steps, size=len(valid_steps), replace=True)
                    boot_meds.append(np.median(b))
                step_lower = np.percentile(boot_meds, 2.5)
                step_upper = np.percentile(boot_meds, 97.5)
                step_ci_str = f"[{int(step_lower)}, {int(step_upper)}]"
            else:
                step_ci_str = "[- , -]"

            med_str = f"{int(med_step)}"
        else:
            med_str = "N/A"
            step_ci_str = "N/A"

        # Calculate Effect size vs pure
        effect_str = "-"
        if cond != "pure" and len(valid_steps) > 1 and len(pure_steps_valid) > 1:
            d, lower, upper = bootstrap_effect_size(np.array(pure_steps_valid), np.array(valid_steps))
            effect_str = f"{d:.2f} [{lower:.2f}, {upper:.2f}]"
        elif cond == "pure":
            effect_str = "0.00"

        lines.append(f"| {cond} | {med_str} | {step_ci_str} | {n_seeds} | {p_str} | {p_ci_str} | {effect_str} |")

    return "\n".join(lines)

if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/multi_seed")
    print(generate_multi_seed_report(results_dir))
