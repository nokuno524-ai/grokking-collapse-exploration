import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse

# --- Styling Settings for Publication ---
# Single column width: 3.25 inches
# Double column width: 6.75 inches
# Colorblind-friendly palette
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def set_style():
    """Sets matplotlib styling for publication quality."""
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)

    # Use LaTeX fonts if available (requires latex installed, falling back if not)
    mpl.rcParams['font.family'] = 'serif'
    # Fallback fonts if not strictly using tex
    mpl.rcParams['font.serif'] = ['Times', 'Times New Roman', 'Computer Modern Roman']

    # Optional: uncomment if system has LaTeX fully installed and configured
    # mpl.rcParams['text.usetex'] = True

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['figure.titlesize'] = 13
    mpl.rcParams['lines.linewidth'] = 1.5

def get_figure(width='single', aspect_ratio=0.75):
    """Returns a matplotlib figure with appropriate dimensions."""
    if width == 'single':
        w = 3.25
    elif width == 'double':
        w = 6.75
    else:
        w = 3.25

    h = w * aspect_ratio
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax

def save_fig(fig, base_path, name):
    """Saves figure in both PDF and PNG formats."""
    os.makedirs(base_path, exist_ok=True)
    fig.savefig(os.path.join(base_path, f"{name}.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(base_path, f"{name}.png"), bbox_inches='tight', dpi=300)

def load_result(filepath):
    """Loads a single result JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_all_results(base_dir):
    """Loads all results and groups them by condition."""
    results = {}
    for root, _, files in os.walk(base_dir):
        if "results.json" in files:
            path = os.path.join(root, "results.json")
            res = load_result(path)

            cfg = res.get("config", {})
            data_cfg = cfg.get("data", {})
            train_cfg = cfg.get("training", {})

            # Create a unique key for the condition, ignoring seed
            c_ratio = data_cfg.get("collapse_ratio", 0.0)
            n_frac = data_cfg.get("noise_fraction", 0.0)
            wd = train_cfg.get("weight_decay", 1.0)

            condition_key = f"c{c_ratio}_n{n_frac}_wd{wd}"

            if condition_key not in results:
                results[condition_key] = {
                    "config": {"collapse_ratio": c_ratio, "noise_fraction": n_frac, "weight_decay": wd},
                    "runs": []
                }
            results[condition_key]["runs"].append(res)

    return results

def plot_grokking_curves(results, output_dir):
    """Plots training and validation accuracy vs steps for all conditions."""
    fig, ax = get_figure(width='double', aspect_ratio=0.5)

    for i, (key, condition_data) in enumerate(results.items()):
        color = CB_color_cycle[i % len(CB_color_cycle)]
        label = f"c={condition_data['config']['collapse_ratio']}, n={condition_data['config']['noise_fraction']}"

        # Aggregate history across runs
        all_train_acc = []
        all_test_acc = []
        steps = None

        for run in condition_data["runs"]:
            if not run.get("history"):
                continue
            history = run["history"]
            if steps is None:
                steps = [entry["step"] for entry in history]

            # Align by step just in case
            run_steps = [entry["step"] for entry in history]
            # Interpolate or match if needed, here we assume uniform eval intervals
            if len(run_steps) == len(steps) and run_steps == steps:
                all_train_acc.append([entry.get("train_acc", 0) for entry in history])
                all_test_acc.append([entry.get("test_acc", 0) for entry in history])

        if not steps or not all_train_acc:
            continue

        train_acc_mean = np.mean(all_train_acc, axis=0)
        test_acc_mean = np.mean(all_test_acc, axis=0)

        ax.plot(steps, train_acc_mean, color=color, linestyle='--', alpha=0.5)
        ax.plot(steps, test_acc_mean, color=color, label=label, linewidth=2.0)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Grokking Curves Across Conditions")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize='small')

    save_fig(fig, output_dir, "grokking_curves")
    plt.close(fig)

def plot_weight_norms(results, output_dir):
    """Plots weight norm trajectories vs steps."""
    fig, ax = get_figure(width='single')

    for i, (key, condition_data) in enumerate(results.items()):
        color = CB_color_cycle[i % len(CB_color_cycle)]
        label = f"c={condition_data['config']['collapse_ratio']}, n={condition_data['config']['noise_fraction']}"

        all_wn = []
        steps = None

        for run in condition_data["runs"]:
            if not run.get("history"):
                continue
            history = run["history"]
            if steps is None:
                steps = [entry["step"] for entry in history]

            if len([entry["step"] for entry in history]) == len(steps):
                all_wn.append([entry.get("weight_norm", 0) for entry in history])

        if not steps or not all_wn:
            continue

        wn_mean = np.mean(all_wn, axis=0)
        wn_std = np.std(all_wn, axis=0)

        ax.plot(steps, wn_mean, color=color, label=label)
        ax.fill_between(steps, wn_mean - wn_std, wn_mean + wn_std, color=color, alpha=0.2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Weight Norm (L2)")
    ax.set_title("Weight Norm Trajectories")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')

    save_fig(fig, output_dir, "weight_norms")
    plt.close(fig)

def plot_loss_landscapes(results, output_dir):
    """Plots training vs test loss (proxy for landscape exploration)."""
    fig, ax = get_figure(width='single')

    for i, (key, condition_data) in enumerate(results.items()):
        color = CB_color_cycle[i % len(CB_color_cycle)]
        label = f"c={condition_data['config']['collapse_ratio']}"

        for run_idx, run in enumerate(condition_data["runs"][:1]): # Plot one representative run per condition
            if not run.get("history"):
                continue

            train_losses = [entry.get("train_loss", 0) for entry in run["history"]]
            test_losses = [entry.get("test_loss", 0) for entry in run["history"]]

            ax.scatter(train_losses, test_losses, color=color, alpha=0.3, s=5, label=label if run_idx == 0 else None)

    ax.set_xlabel("Train Loss")
    ax.set_ylabel("Test Loss")
    ax.set_title("Loss Trajectories")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')

    save_fig(fig, output_dir, "loss_trajectories")
    plt.close(fig)

def plot_fourier_concentration(results, output_dir):
    """Plots Fourier concentration (surrogate for attention pattern evolution)."""
    fig, ax = get_figure(width='single')

    for i, (key, condition_data) in enumerate(results.items()):
        color = CB_color_cycle[i % len(CB_color_cycle)]
        label = f"c={condition_data['config']['collapse_ratio']}, n={condition_data['config']['noise_fraction']}"

        all_fc = []
        steps = None

        for run in condition_data["runs"]:
            if not run.get("history"):
                continue
            history = run["history"]
            if steps is None:
                steps = [entry["step"] for entry in history]

            if len([entry["step"] for entry in history]) == len(steps):
                all_fc.append([entry.get("fourier_concentration", 0) for entry in history])

        if not steps or not all_fc:
            continue

        fc_mean = np.mean(all_fc, axis=0)

        ax.plot(steps, fc_mean, color=color, label=label)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Fourier Concentration")
    ax.set_title("Embedding Evolution")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')

    save_fig(fig, output_dir, "fourier_concentration")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Paper Plots")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with experiment results")
    parser.add_argument("--output_dir", type=str, default="plots/output", help="Directory to save plots")
    args = parser.parse_args()

    set_style()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} distinct conditions. Generating plots...")

    plot_grokking_curves(results, args.output_dir)
    plot_weight_norms(results, args.output_dir)
    plot_loss_landscapes(results, args.output_dir)
    plot_fourier_concentration(results, args.output_dir)

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
