import argparse
import os
import json
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import TrainConfig
from train import train

def run_sweep(output_dir: str, max_steps: int):
    # To cover more parameters we include lr and d_model sweeps
    collapse_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    d_models = [64, 128, 256]
    training_durations = [1000, 5000, max_steps]

    results = {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # We will do a small sweep for demo purposes over collapse level and LR to generate 2D plots
    for cl, lr in itertools.product(collapse_levels, lrs):
        cond_name = f"cl_{cl:.1f}_lr_{lr:.4f}"
        print(f"Running sweep: {cond_name}")
        config = TrainConfig(
            collapse_level=cl,
            lr=lr,
            condition_name=cond_name,
            output_dir=output_dir,
            max_steps=max_steps,
            eval_every=max_steps // 5,  # Fewer evals for speed during sweep
            log_every=max_steps // 2
        )
        state = train(config)
        results[cond_name] = {
            "collapse_level": cl,
            "lr": lr,
            "d_model": 128,  # Default
            "max_steps": max_steps,
            "grokked": state.grokked,
            "grokking_step": state.grokking_step,
            "test_acc": state.test_acc,
            "train_acc": state.train_acc
        }

    with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

def plot_phase_diagram(results_path: str, output_path: str):
    with open(results_path, "r") as f:
        results = json.load(f)

    cls = sorted(list(set([v["collapse_level"] for v in results.values()])))
    lrs = sorted(list(set([v["lr"] for v in results.values()])))

    matrix = np.zeros((len(lrs), len(cls)))

    for k, v in results.items():
        i = lrs.index(v["lr"])
        j = cls.index(v["collapse_level"])

        # 2: Grokked
        # 1: Memorization (high train, low test)
        # 0: Collapse/Failed to learn (low train, low test)
        if v["grokked"]:
            matrix[i, j] = 2
        elif v["train_acc"] > 0.9 and v["test_acc"] < 0.9:
            matrix[i, j] = 1
        else:
            matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap='viridis', origin='lower')

    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels([f"{c:.1f}" for c in cls])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"{w:.4f}" for w in lrs])

    ax.set_xlabel('Collapse Level (Synthetic Data Ratio)')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Phase Diagram: Grokking vs Memorization vs Collapse')

    cbar = fig.colorbar(cax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Failed/Collapse', 'Memorization', 'Grokking'])

    plt.savefig(output_path)
    print(f"Saved phase diagram to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="sweep_results")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep(args.output_dir, args.max_steps)

    plot_phase_diagram(os.path.join(args.output_dir, "sweep_results.json"), os.path.join(args.output_dir, "phase_diagram.png"))
