import json
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

def calculate_time_to_grok(history: List[dict], grok_threshold: float = 0.95, cleanup_step: int = 0) -> int:
    """Calculate the number of steps it takes to grok after cleanup."""
    for entry in history:
        if entry["step"] >= cleanup_step and entry["test_acc"] >= grok_threshold:
            return entry["step"] - cleanup_step
    return -1

def calculate_grok_probability(results_dir: str, grok_threshold: float = 0.95, cleanup_step: int = 0) -> float:
    """Calculate the fraction of seeds that grokked after the cleanup step."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return 0.0

    seeds_grokked = 0
    total_seeds = 0
    for seed_dir in results_path.iterdir():
        if seed_dir.is_dir():
            res_file = seed_dir / "results.json"
            if res_file.exists():
                total_seeds += 1
                try:
                    with open(res_file, "r") as f:
                        data = json.load(f)
                        ttg = calculate_time_to_grok(data.get("history", []), grok_threshold, cleanup_step)
                        if ttg >= 0:
                            seeds_grokked += 1
                except Exception as e:
                    print(f"Error reading {res_file}: {e}")

    return seeds_grokked / total_seeds if total_seeds > 0 else 0.0

def calculate_survival_curve(histories: List[List[dict]], grok_threshold: float = 0.95, cleanup_step: int = 0) -> Tuple[List[int], List[float]]:
    """
    Calculate the survival curve: fraction of seeds remaining un-grokked vs steps since cleanup.
    Returns (steps, survival_fraction)
    """
    if not histories:
        return [], []

    ttg_list = []
    for h in histories:
        ttg = calculate_time_to_grok(h, grok_threshold, cleanup_step)
        if ttg >= 0:
            ttg_list.append(ttg)

    # max steps evaluated across all histories after cleanup
    max_steps = max((h[-1]["step"] - cleanup_step for h in histories if h and h[-1]["step"] >= cleanup_step), default=0)
    if max_steps <= 0:
        return [], []

    steps = sorted(list(set(entry["step"] - cleanup_step for h in histories for entry in h if entry["step"] >= cleanup_step)))
    survival_fraction = []

    total_seeds = len(histories)
    for step in steps:
        survived = sum(1 for ttg in ttg_list if ttg > step) + (total_seeds - len(ttg_list))
        survival_fraction.append(survived / total_seeds)

    return steps, survival_fraction

def plot_survival_curves(results_dirs: Dict[str, str], output_path: str, grok_threshold: float = 0.95, cleanup_step: int = 0):
    """Plot survival curves for multiple conditions."""
    plt.figure(figsize=(10, 6))

    for label, r_dir in results_dirs.items():
        results_path = Path(r_dir)
        histories = []
        if results_path.exists():
            for seed_dir in results_path.iterdir():
                if seed_dir.is_dir():
                    res_file = seed_dir / "results.json"
                    if res_file.exists():
                        try:
                            with open(res_file, "r") as f:
                                data = json.load(f)
                                histories.append(data.get("history", []))
                        except Exception as e:
                            print(f"Error reading {res_file}: {e}")

        if histories:
            steps, sf = calculate_survival_curve(histories, grok_threshold, cleanup_step)
            plt.plot(steps, sf, label=label, marker='o', markersize=4)

    plt.xlabel(f"Steps since cleanup (step {cleanup_step})")
    plt.ylabel("Fraction of seeds un-grokked (Survival)")
    plt.title("Grokking Survival Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved survival curve to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-step", type=int, default=0, help="Step where pure data is restored")
    args = parser.parse_args()
    print("Metrics script ready.")
