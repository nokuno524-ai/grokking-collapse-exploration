import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

def load_data(results_dir: Path):
    data = {}
    for condition in SEVERITY_ORDER:
        json_path = results_dir / condition / "results.json"
        if json_path.exists():
            with open(json_path) as f:
                data[condition] = json.load(f)
    return data

def create_dashboard(results_dir: str = "results", output_dir: str = "viz"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = load_data(results_path)
    if not data:
        print("No data found.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    # Panel A: Test Accuracy
    ax = axs[0]
    for cond, d in data.items():
        if "history" in d:
            steps = [e["step"] for e in d["history"]]
            acc = [e.get("test_acc", 0) for e in d["history"]]
            ax.plot(steps, acc, label=cond.replace("_", " ").title(), color=colors.get(cond, "black"))
    ax.set_title("Panel A: Test Accuracy over Training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Test Accuracy")
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='Grokking Threshold (95%)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B: Weight Norm
    ax = axs[1]
    for cond, d in data.items():
        if "history" in d:
            steps = [e["step"] for e in d["history"]]
            norms = [e.get("weight_norm", 0) for e in d["history"]]
            ax.plot(steps, norms, label=cond.replace("_", " ").title(), color=colors.get(cond, "black"))
    ax.set_title("Panel B: Weight Norm Trajectories")
    ax.set_xlabel("Step")
    ax.set_ylabel("Weight Norm (L2)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel C: Loss Landscape proxy (Test Loss vs Train Loss)
    ax = axs[2]
    for cond, d in data.items():
        if "history" in d:
            train_loss = [e.get("train_loss", 0) for e in d["history"]]
            test_loss = [e.get("test_loss", 0) for e in d["history"]]
            ax.plot(train_loss, test_loss, alpha=0.6, label=cond.replace("_", " ").title(), color=colors.get(cond, "black"))
            if len(train_loss) > 0:
                ax.scatter(train_loss[-1], test_loss[-1], color=colors.get(cond, "black"), marker='x', s=100)
    ax.set_title("Panel C: Training Trajectories (Test Loss vs Train Loss)")
    ax.set_xlabel("Train Loss")
    ax.set_ylabel("Test Loss")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel D: Grokking Onset Steps
    ax = axs[3]
    labels = []
    steps_onset = []
    bar_colors = []

    for cond in SEVERITY_ORDER:
        if cond in data:
            d = data[cond]
            labels.append(cond.replace("_", "\n").title())
            bar_colors.append(colors.get(cond, "black"))
            g_step = d.get("grokking_step")
            if g_step is None or g_step == 0:
                # Use max step to indicate failure to grok, or inf proxy
                max_step = d["config"].get("max_steps", 50000)
                steps_onset.append(max_step)
            else:
                steps_onset.append(g_step)

    bars = ax.bar(labels, steps_onset, color=bar_colors)
    ax.set_title("Panel D: Grokking Onset Step")
    ax.set_ylabel("Step")
    ax.set_ylim(0, 55000)

    # Annotate bars
    for bar, d_key in zip(bars, [c for c in SEVERITY_ORDER if c in data]):
        yval = bar.get_height()
        d = data[d_key]
        g_step = d.get("grokking_step")
        text = str(g_step) if g_step else "Failed\nto grok"
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1000, text, ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "results_dashboard.png", dpi=300)
    plt.savefig(out_path / "results_dashboard.pdf", dpi=300)
    print(f"Dashboard saved to {out_path}/results_dashboard.png and .pdf")

if __name__ == "__main__":
    create_dashboard()
