import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Apply publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
})

RESULTS_DIR = Path('results')
MULTI_SEED_DIR = RESULTS_DIR / 'multi_seed'
CONDITIONS = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

COLORS = {
    "pure": "#2ecc71",
    "low_collapse": "#3498db",
    "medium_collapse": "#f39c12",
    "high_collapse": "#e74c3c",
    "severe_collapse": "#8e44ad",
}

LABELS = {
    "pure": "Pure Data (0% Collapse)",
    "low_collapse": "Low Collapse (5% Noise)",
    "medium_collapse": "Medium Collapse (10% Noise)",
    "high_collapse": "High Collapse (15% Noise)",
    "severe_collapse": "Severe Collapse (20% Noise)",
}

def load_multi_seed_data(condition, metric):
    seeds = [42, 43, 44, 45, 46]
    all_values = []
    steps = None
    for seed in seeds:
        path = MULTI_SEED_DIR / str(seed) / condition / 'results.json'
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
            history = data.get('history', [])
            if not history:
                continue
            if steps is None:
                steps = [entry['step'] for entry in history]
            values = [entry.get(metric, 0) for entry in history]
            all_values.append(values)

    if not all_values:
        return None, None, None, None

    all_values = np.array(all_values)
    mean_vals = np.mean(all_values, axis=0)
    std_vals = np.std(all_values, axis=0)

    # 95% Confidence Interval (approx)
    ci = 1.96 * std_vals / np.sqrt(len(seeds))
    return np.array(steps), mean_vals, mean_vals - ci, mean_vals + ci

def load_multi_seed_layer_data(condition, metric_prefix="weight_norm"):
    # Since this repo uses a 1-layer transformer and tracks total weight_norm,
    # we simulate or map "per layer" appropriately if it was tracked.
    # We will just plot the available metric here.
    return load_multi_seed_data(condition, metric_prefix)

def plot_fig1():
    print("Generating Figure 1 (Accuracy Curves)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for cond in CONDITIONS:
        steps, mean_train, lower_train, upper_train = load_multi_seed_data(cond, 'train_acc')
        _, mean_test, lower_test, upper_test = load_multi_seed_data(cond, 'test_acc')

        if steps is None:
            continue

        ax1.plot(steps, mean_train, label=LABELS[cond], color=COLORS[cond])
        ax1.fill_between(steps, lower_train, upper_train, color=COLORS[cond], alpha=0.2)

        ax2.plot(steps, mean_test, label=LABELS[cond], color=COLORS[cond])
        ax2.fill_between(steps, lower_test, upper_test, color=COLORS[cond], alpha=0.2)

    ax1.set_title('Training Accuracy vs. Steps')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Test Accuracy vs. Steps (Grokking)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.95, color='black', linestyle='--', alpha=0.5, label='95% Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('results/figure1_accuracy.pdf')
    plt.savefig('results/figure1_accuracy.png', dpi=300)
    plt.close()

def plot_fig2():
    print("Generating Figure 2 (Weight Norms Per Layer)...")
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond in CONDITIONS:
        # Note: The codebase currently aggregates weight norm for the whole model in `weight_norm`.
        # Since it's a 1-layer transformer, this effectively represents the single layer's norm.
        steps, mean_val, lower, upper = load_multi_seed_layer_data(cond, 'weight_norm')

        if steps is None:
            continue

        ax.plot(steps, mean_val, label=LABELS[cond], color=COLORS[cond])
        ax.fill_between(steps, lower, upper, color=COLORS[cond], alpha=0.2)

    ax.set_title('Weight Norm Evolution (Per Layer/Total)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('results/figure2_weight_norm.pdf')
    plt.savefig('results/figure2_weight_norm.png', dpi=300)
    plt.close()

def plot_fig3():
    print("Generating Figure 3 (Phase Diagram from Grid)...")
    import re

    grid_dir = RESULTS_DIR / 'grid'
    if not grid_dir.exists():
        print("No grid data found for phase diagram.")
        return

    data_points = []
    for path in grid_dir.glob('*/seed_42/results.json'):
        dir_name = path.parent.parent.name
        match = re.match(r'level([\d.]+)_sev([\d.]+)', dir_name)
        if match:
            level = float(match.group(1))
            sev = float(match.group(2))
            with open(path) as f:
                data = json.load(f)
                grok = data.get('grokking_step', None)
                data_points.append({'level': level, 'severity': sev, 'grokking_step': grok})

    if not data_points:
        return

    import pandas as pd
    df = pd.DataFrame(data_points)
    pivot = df.pivot_table(index='level', columns='severity', values='grokking_step')

    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap='viridis_r', ax=ax, cbar_kws={'label': 'Grokking Step'})

    ax.set_title('Phase Diagram: Grokking Step vs. Collapse Parameters')
    ax.set_ylabel('Collapse Level')
    ax.set_xlabel('Collapse Severity')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('results/figure3_phase_diagram.pdf')
    plt.savefig('results/figure3_phase_diagram.png', dpi=300)
    plt.close()

def plot_fig4():
    print("Generating Figure 4 (Fourier Concentration)...")
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond in CONDITIONS:
        steps, mean_val, lower, upper = load_multi_seed_data(cond, 'fourier_concentration')

        if steps is None:
            continue

        ax.plot(steps, mean_val, label=LABELS[cond], color=COLORS[cond])
        ax.fill_between(steps, lower, upper, color=COLORS[cond], alpha=0.2)

    ax.set_title('Fourier Concentration Evolution')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Concentration')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('results/figure4_fourier.pdf')
    plt.savefig('results/figure4_fourier.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    print("Done!")
