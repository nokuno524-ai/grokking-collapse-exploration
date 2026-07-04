import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path

def plot_multi_panel_summary(results_dir="results", save_dir="visualizations/outputs"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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
        train_acc = [entry["train_acc"] for entry in history]
        test_acc = [entry["test_acc"] for entry in history]
        fourier = [entry["fourier_concentration"] for entry in history]

        # Plot Test Acc
        axes[0].plot(steps, test_acc, label=run_path.name)

        # Plot Train Acc
        axes[1].plot(steps, train_acc, label=run_path.name)

        # Plot Fourier
        axes[2].plot(steps, fourier, label=run_path.name)

    axes[0].set_title("Test Accuracy")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Train Accuracy")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Fourier Concentration")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Concentration")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(save_dir) / "multi_panel_summary.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    plot_multi_panel_summary(args.results_dir)
