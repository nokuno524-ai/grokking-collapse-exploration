import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.plot_utils import set_style

def generate_cliff_figure(registry_path: Path, output_dir: Path):
    set_style()

    with open(registry_path) as f:
        registry = json.load(f)

    # We want to plot Noise Fraction vs Grokking Step for weight decay 1.0 and 0.3
    # Use runs from exp_c_grid

    grid_runs = [r for r in registry if "exp_c_grid" in r.get("run_path", "") and r.get("train_fraction") == 0.3]

    # Group by wd and noise
    wds = sorted(list(set([r["weight_decay"] for r in grid_runs])))

    fig, ax = plt.subplots(figsize=(6, 4))

    markers = {1.0: 'o', 0.3: 's', 3.0: '^'}
    colors = {1.0: '#3498db', 0.3: '#2ecc71', 3.0: '#e74c3c'}

    for wd in wds:
        wd_runs = [r for r in grid_runs if r["weight_decay"] == wd]
        noises = sorted(list(set([r["noise_fraction"] for r in wd_runs])))

        grok_rates = []
        mean_grok_steps = []

        plot_noises = []

        for n in noises:
            n_runs = [r for r in wd_runs if r["noise_fraction"] == n]
            if not n_runs:
                continue

            grokked_count = sum(1 for r in n_runs if r["grokked"])
            rate = grokked_count / len(n_runs)

            grok_rates.append(rate)
            plot_noises.append(n)

        if len(plot_noises) > 0:
            ax.plot(plot_noises, grok_rates, marker=markers.get(wd, 'o'),
                    color=colors.get(wd, '#95a5a6'), label=f"Weight Decay = {wd}",
                    linewidth=2, markersize=8)

    ax.set_title("The Grokking Cliff")
    ax.set_xlabel("Label Noise / Contamination Fraction")
    ax.set_ylabel("Grokking Probability (across seeds)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    ax.legend(loc='best')

    plt.tight_layout()
    out_path = output_dir / "fig3_cliff.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_cliff_figure(Path("results/registry.json"), Path("paper/figures"))
