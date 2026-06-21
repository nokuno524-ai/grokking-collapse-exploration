import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def load_training_curves(results_dir="results"):
    """
    Parses results.json files to extract training/test metrics over steps.
    Format: { condition: {"steps": [], "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []} }
    """
    data = {}
    pattern = os.path.join(results_dir, "*/results.json")
    files = glob.glob(pattern)

    for fpath in files:
        condition = os.path.basename(os.path.dirname(fpath))
        with open(fpath, "r") as f:
            res = json.load(f)

        if "history" in res:
            history = res["history"]
            data[condition] = {
                "steps": [h["step"] for h in history],
                "train_acc": [h["train_acc"] for h in history],
                "test_acc": [h["test_acc"] for h in history],
                "train_loss": [h["train_loss"] for h in history],
                "test_loss": [h["test_loss"] for h in history],
            }

    return data

def plot_curves(data, save_path):
    """
    Plots a 1x2 grid of training curves:
    Left: Training & Test Loss
    Right: Training & Test Accuracy
    Since there are multiple conditions, we'll plot them all, colored by condition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.set_style("whitegrid")

    # We will just use solid for train, dashed for test
    for condition, metrics in data.items():
        steps = metrics["steps"]

        # Loss
        axes[0].plot(steps, metrics["train_loss"], label=f"{condition} (Train)", linestyle='-', alpha=0.7)
        axes[0].plot(steps, metrics["test_loss"], label=f"{condition} (Test)", linestyle='--')

        # Accuracy
        axes[1].plot(steps, metrics["train_acc"], label=f"{condition} (Train)", linestyle='-', alpha=0.7)
        axes[1].plot(steps, metrics["test_acc"], label=f"{condition} (Test)", linestyle='--')

    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.05, 1.05)

    # Put legend outside
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import numpy as np
    dummy_data = {
        "pure": {
            "steps": list(range(100, 2000, 100)),
            "train_acc": np.linspace(0.1, 1.0, 19).tolist(),
            "test_acc": [0.0] * 13 + np.linspace(0.1, 1.0, 6).tolist(),
            "train_loss": np.linspace(3.0, 0.01, 19).tolist(),
            "test_loss": [3.0] * 13 + np.linspace(3.0, 0.01, 6).tolist(),
        }
    }

    parsed = load_training_curves()
    if not parsed:
        print("No results found, using dummy data.")
        parsed = dummy_data

    plot_curves(parsed, "visualizations/training_curves.png")
    print("Saved training curves plot to visualizations/training_curves.png")
