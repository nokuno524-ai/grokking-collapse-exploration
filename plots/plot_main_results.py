import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# ICML styling
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'text.usetex': False, # Set to False for compatibility, enable if LaTeX is installed
    'axes.labelsize': 12,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 4),
    'figure.dpi': 300,
})

COLORS = {
    "pure": "#0072B2",           # Blue
    "low_collapse": "#56B4E9",   # Light Blue
    "medium_collapse": "#009E73",# Green
    "high_collapse": "#E69F00",  # Orange
    "severe_collapse": "#D55E00" # Red
}

LABELS = {
    "pure": "Pure (0%)",
    "low_collapse": "Low (5%)",
    "medium_collapse": "Medium (15%)",
    "high_collapse": "High (30%)",
    "severe_collapse": "Severe (50%)"
}

def plot_grokking_curves(results_dir: Path, output_file: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    for c in conditions:
        cond_dir = results_dir / c
        if not cond_dir.exists():
            continue

        with open(cond_dir / "results.json") as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            continue

        steps = [e["step"] for e in history]
        test_acc = [e.get("test_acc", 0) for e in history]
        fourier = [e.get("fourier_concentration", 0) for e in history]

        ax1.plot(steps, test_acc, label=LABELS[c], color=COLORS[c], linewidth=2)
        ax2.plot(steps, fourier, label=LABELS[c], color=COLORS[c], linewidth=2)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Generalization Delay")
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Fourier Concentration")
    ax2.set_title("Circuit Formation")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved {output_file}")

if __name__ == "__main__":
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_grokking_curves(Path("results"), out_dir / "main_grokking_curves.pdf")
