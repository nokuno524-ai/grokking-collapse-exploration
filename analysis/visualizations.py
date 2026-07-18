import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics(metrics_path="analysis/attention_metrics.json"):
    """Load the computed attention metrics."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_attention_heatmap(metrics, condition="pure", step_idx=-1, output_path="analysis/plots/attention_heatmap.png"):
    """
    Plot attention head heatmaps for a specific condition and step.

    Args:
        metrics: Dictionary of metrics loaded from JSON
        condition: Which condition to plot (e.g. 'pure')
        step_idx: Index of the step to plot (-1 for final step)
        output_path: Where to save the plot
    """
    if condition not in metrics or not metrics[condition]:
        print(f"No data for condition: {condition}")
        return

    data = metrics[condition][step_idx]
    step = data['step']
    mean_attn = np.array(data['mean_attn_matrix']) # (n_heads, 2, 2)
    n_heads = mean_attn.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        sns.heatmap(mean_attn[h], ax=axes[h], annot=True, cmap="YlGnBu",
                    vmin=0, vmax=1, fmt=".2f",
                    xticklabels=['pos 1', 'pos 2'],
                    yticklabels=['pos 1', 'pos 2'])
        axes[h].set_title(f"Head {h+1}")
        axes[h].set_ylabel("Query")
        axes[h].set_xlabel("Key")

    plt.suptitle(f"Mean Attention Weights - {condition.title()} - Step {step}")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_evolution(metrics, output_path="analysis/plots/entropy_evolution.png"):
    """
    Plot the evolution of total attention entropy over training for all conditions.
    """
    plt.figure(figsize=(10, 6))

    colors = {
        'pure': 'blue',
        'low_collapse': 'green',
        'medium_collapse': 'orange',
        'severe_collapse': 'red',
        'high_collapse': 'purple'
    }

    for condition, data_list in metrics.items():
        if not data_list:
            continue

        steps = [d['step'] for d in data_list]
        entropies = [d['entropy_total'] for d in data_list]

        plt.plot(steps, entropies, label=condition,
                 color=colors.get(condition, 'black'),
                 linewidth=2)

    plt.title("Attention Entropy Evolution")
    plt.xlabel("Training Step")
    plt.ylabel("Mean Attention Entropy (nats)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_circuit_formation(metrics, condition="pure", output_path="analysis/plots/circuit_formation.png"):
    """
    Plot the evolution of self-attention vs cross-attention over time to show circuit formation.
    """
    if condition not in metrics or not metrics[condition]:
        return

    data_list = metrics[condition]
    steps = [d['step'] for d in data_list]

    # Extract self-attention and cross-attention scores for all heads
    # Shape: (num_steps, n_heads)
    self_attn = np.array([d['circuits']['self_attention_score'] for d in data_list])
    cross_attn = np.array([d['circuits']['cross_attention_score'] for d in data_list])

    n_heads = self_attn.shape[1]

    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        axes[h].plot(steps, self_attn[:, h], label="Self Attention", color="blue")
        axes[h].plot(steps, cross_attn[:, h], label="Cross Attention", color="red")
        axes[h].set_title(f"Head {h+1}")
        axes[h].set_xlabel("Training Step")
        axes[h].set_ylabel("Probability")
        axes[h].set_ylim(0, 1.05)
        axes[h].grid(True, alpha=0.3)
        if h == 0:
            axes[h].legend()

    plt.suptitle(f"Circuit Formation - {condition.title()}")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_condition_comparisons(metrics, output_dir="analysis/plots"):
    """
    Generate side-by-side comparisons of the final attention state across conditions.
    """
    conditions = [c for c in ['pure', 'low_collapse', 'medium_collapse', 'severe_collapse', 'high_collapse']
                 if c in metrics and metrics[c]]

    if not conditions:
        return

    # Get final step metrics for each condition
    final_metrics = {c: metrics[c][-1] for c in conditions}

    # 1. Compare total entropy
    plt.figure(figsize=(8, 5))
    entropies = [final_metrics[c]['entropy_total'] for c in conditions]
    sns.barplot(x=conditions, y=entropies, palette="viridis")
    plt.title("Final Attention Entropy by Condition")
    plt.ylabel("Mean Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_comparison.png"), dpi=300)
    plt.close()

    # 2. Compare circuit dominance (max cross attention across heads)
    plt.figure(figsize=(8, 5))
    # For each condition, find the head with the strongest cross-attention
    max_cross = [np.max(final_metrics[c]['circuits']['cross_attention_score']) for c in conditions]
    sns.barplot(x=conditions, y=max_cross, palette="magma")
    plt.title("Maximum Cross-Attention Score by Condition")
    plt.ylabel("Max Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "circuit_comparison.png"), dpi=300)
    plt.close()

def plot_3d_metrics(metrics, output_path="analysis/plots/entropy_3d.png"):
    """
    Generate a 3D surface plot showing entropy over (Condition × Step).
    """
    conditions = [c for c in ['pure', 'low_collapse', 'medium_collapse', 'severe_collapse', 'high_collapse']
                 if c in metrics and metrics[c]]

    if not conditions:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # We need a meshgrid. Find common steps or interpolate
    # For simplicity, just use the step indices if they roughly align
    all_steps = sorted(list(set([d['step'] for d in metrics['pure']])))

    Y = np.arange(len(conditions))
    X = np.array(all_steps)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X, dtype=float)

    for i, cond in enumerate(conditions):
        cond_data = {d['step']: d['entropy_total'] for d in metrics[cond]}
        for j, step in enumerate(all_steps):
            # Interpolate or forward fill if step missing
            # A simple approach: find closest step
            if step in cond_data:
                Z[i, j] = cond_data[step]
            else:
                closest_step = min(cond_data.keys(), key=lambda k: abs(k - step))
                Z[i, j] = cond_data[closest_step]

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Collapse Level')
    ax.set_zlabel('Attention Entropy')

    # Set y ticks to condition names
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels([c.replace('_collapse', '') for c in conditions])

    plt.title('Attention Entropy Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(metrics_path="analysis/attention_metrics.json"):
    """Generate all visualizations."""
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    metrics = load_metrics(metrics_path)

    # Heatmaps for final step of each condition
    for condition in metrics:
        plot_attention_heatmap(metrics, condition, -1, f"analysis/plots/heatmap_{condition}.png")
        plot_circuit_formation(metrics, condition, f"analysis/plots/circuit_{condition}.png")

    plot_entropy_evolution(metrics)
    plot_condition_comparisons(metrics)
    plot_3d_metrics(metrics)
    print("All visualizations generated in analysis/plots/")

if __name__ == "__main__":
    generate_all_plots()
