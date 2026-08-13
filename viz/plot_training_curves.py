import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_curves(results_dir="results", output_dir="viz/output"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)

    for condition in conditions:
        json_file = results_path / condition / "results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)

            steps = []
            test_acc = []
            grokking_step = None

            for entry in data.get("history", []):
                steps.append(entry["step"])
                test_acc.append(entry["test_acc"])
                if grokking_step is None and entry["test_acc"] >= 0.8:
                    grokking_step = entry["step"]

            plt.plot(steps, test_acc, label=condition)
            if grokking_step is not None:
                plt.axvline(x=grokking_step, linestyle='--', alpha=0.5)

    plt.title("Test Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)

    for condition in conditions:
        json_file = results_path / condition / "results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)

            steps = []
            test_loss = []

            for entry in data.get("history", []):
                steps.append(entry["step"])
                test_loss.append(entry["test_loss"])

            plt.plot(steps, test_loss, label=condition)

    plt.title("Test Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path / "training_curves.png")
    plt.close()

if __name__ == "__main__":
    plot_training_curves()
