import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path

def plot_weight_norm_trajectory(results_dir="results", save_dir="visualizations/outputs"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    run_paths = [p for p in Path(results_dir).iterdir() if p.is_dir()]

    for run_path in run_paths:
        res_file = run_path / "results.json"
        if not res_file.exists():
            continue

        with open(res_file, "r") as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            continue

        steps = [entry["step"] for entry in history]
        norms = [entry["weight_norm"] for entry in history]

        plt.plot(steps, norms, label=run_path.name)

    plt.xlabel("Training Step")
    plt.ylabel("Weight Norm (L2)")
    plt.title("Weight Norm Trajectory across Collapse Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = Path(save_dir) / "weight_norm_trajectory.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    plot_weight_norm_trajectory(args.results_dir)
