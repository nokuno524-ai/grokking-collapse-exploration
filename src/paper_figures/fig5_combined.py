import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.plot_utils import set_style, get_color_palette

def generate_combined_figure(registry_path: Path, output_dir: Path):
    set_style()

    with open(registry_path) as f:
        registry = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = get_color_palette()
    target_conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    # --- Top Left: Test Accuracy (Grokking) ---
    ax1 = axes[0, 0]
    for condition in target_conditions:
        runs = [r for r in registry if r.get("condition_name") == condition and "seed_sweep" in r.get("run_path", "")]
        if not runs:
            runs = [r for r in registry if r.get("condition_name") == condition]
        if runs:
            run = next((r for r in runs if r.get("seed") == 42), runs[0])
            run_path = Path(run["run_path"]) / "results.json"
            try:
                with open(run_path) as f:
                    data = json.load(f)
                history = data.get("history", [])
                steps = [h["step"] for h in history]
                test_acc = [h["test_acc"] for h in history]
                label = condition.replace("_", " ").title()
                ax1.plot(steps, test_acc, color=colors[condition], label=label, linewidth=1.5)
            except:
                pass
    ax1.set_title("A. Test Accuracy (Grokking)")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Accuracy")
    ax1.axhline(0.95, color='gray', linestyle='--', alpha=0.8)

    # --- Top Right: Weight Norm ---
    ax2 = axes[0, 1]
    has_labels = False
    for condition in target_conditions:
        runs = [r for r in registry if r.get("condition_name") == condition and "seed_sweep" in r.get("run_path", "")]
        if not runs:
            runs = [r for r in registry if r.get("condition_name") == condition]
        if runs:
            run = next((r for r in runs if r.get("seed") == 42), runs[0])
            run_path = Path(run["run_path"]) / "results.json"
            try:
                with open(run_path) as f:
                    data = json.load(f)
                history = data.get("history", [])
                valid_steps = [h["step"] for h in history if "weight_norm" in h]
                weight_norm = [h["weight_norm"] for h in history if "weight_norm" in h]
                if weight_norm:
                    label = condition.replace("_", " ").title()
                    ax2.plot(valid_steps, weight_norm, color=colors[condition], label=label, linewidth=1.5)
                    has_labels = True
            except:
                pass
    ax2.set_title("B. Weight Norm Trajectory")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("L2 Norm")
    if has_labels:
        ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # --- Bottom Left: Generalization Gap ---
    ax3 = axes[1, 0]
    for condition in target_conditions:
        runs = [r for r in registry if r.get("condition_name") == condition and "seed_sweep" in r.get("run_path", "")]
        if not runs:
            runs = [r for r in registry if r.get("condition_name") == condition]
        if runs:
            run = next((r for r in runs if r.get("seed") == 42), runs[0])
            run_path = Path(run["run_path"]) / "results.json"
            try:
                with open(run_path) as f:
                    data = json.load(f)
                history = data.get("history", [])
                steps = [h["step"] for h in history]
                train_loss = np.array([h["train_loss"] for h in history])
                test_loss = np.array([h["test_loss"] for h in history])
                gap = test_loss - train_loss
                label = condition.replace("_", " ").title()
                ax3.plot(steps, gap, color=colors[condition], label=label, linewidth=1.5)
            except:
                pass
    ax3.set_title("C. Generalization Gap (Test - Train Loss)")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Loss Gap")

    # --- Bottom Right: The Grokking Cliff ---
    ax4 = axes[1, 1]
    grid_runs = [r for r in registry if "exp_c_grid" in r.get("run_path", "") and r.get("train_fraction") == 0.3]
    wds = sorted(list(set([r["weight_decay"] for r in grid_runs])))
    markers = {1.0: 'o', 0.3: 's', 3.0: '^'}
    cliff_colors = {1.0: '#3498db', 0.3: '#2ecc71', 3.0: '#e74c3c'}

    has_labels = False
    for wd in wds:
        wd_runs = [r for r in grid_runs if r["weight_decay"] == wd]
        noises = sorted(list(set([r["noise_fraction"] for r in wd_runs])))
        grok_rates = []
        plot_noises = []
        for n in noises:
            n_runs = [r for r in wd_runs if r["noise_fraction"] == n]
            if not n_runs:
                continue
            rate = sum(1 for r in n_runs if r["grokked"]) / len(n_runs)
            grok_rates.append(rate)
            plot_noises.append(n)
        if plot_noises:
            ax4.plot(plot_noises, grok_rates, marker=markers.get(wd, 'o'),
                    color=cliff_colors.get(wd, '#95a5a6'), label=f"Weight Decay = {wd}",
                    linewidth=2, markersize=8)
            has_labels = True

    ax4.set_title("D. The Grokking Cliff")
    ax4.set_xlabel("Label Noise / Contamination Fraction")
    ax4.set_ylabel("Grokking Probability")
    ax4.set_ylim(-0.05, 1.05)
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    if has_labels:
        ax4.legend(loc='best')

    plt.tight_layout()
    out_path = output_dir / "fig5_combined.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_combined_figure(Path("results/registry.json"), Path("paper/figures"))
