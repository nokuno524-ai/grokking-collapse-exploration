import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

def draw_setup_figure(output_path):
    """
    Draw experimental setup (data pipeline, architecture) using matplotlib.
    """
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Hide axes
    ax.axis('off')

    # Data Pipeline block
    data_rect = patches.Rectangle((0.1, 0.6), 0.3, 0.3, linewidth=2, edgecolor='black', facecolor='#e6f2ff')
    ax.add_patch(data_rect)
    ax.text(0.25, 0.85, 'Data Pipeline', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0.25, 0.75, 'Clean Data:\n(a + b) mod p', ha='center', va='center')
    ax.text(0.25, 0.65, 'Collapse Contamination:\nTemperature warped', ha='center', va='center')

    # Arrow
    ax.arrow(0.4, 0.75, 0.1, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')

    # Model Architecture block
    model_rect = patches.Rectangle((0.55, 0.4), 0.4, 0.5, linewidth=2, edgecolor='black', facecolor='#fff2e6')
    ax.add_patch(model_rect)
    ax.text(0.75, 0.85, 'ModularArithmeticTransformer', ha='center', va='center', fontweight='bold', fontsize=12)

    # Inner model details
    ax.text(0.75, 0.75, 'Token & Pos Embeddings', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.arrow(0.75, 0.72, 0, -0.04, head_width=0.02, head_length=0.02, fc='k', ec='k')
    ax.text(0.75, 0.63, '1-layer Transformer\n(d_model=128, n_heads=4)', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.arrow(0.75, 0.58, 0, -0.04, head_width=0.02, head_length=0.02, fc='k', ec='k')
    ax.text(0.75, 0.50, 'Mean Pool & Output Head', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.arrow(0.75, 0.47, 0, -0.04, head_width=0.02, head_length=0.02, fc='k', ec='k')
    ax.text(0.75, 0.42, 'Logits (p classes)', ha='center', va='center')

    # Grokking block
    grok_rect = patches.Rectangle((0.1, 0.1), 0.85, 0.2, linewidth=2, edgecolor='black', facecolor='#f2ffe6')
    ax.add_patch(grok_rect)
    ax.text(0.525, 0.25, 'Grokking Evaluation', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0.525, 0.18, 'Measure grokking step (test acc > 0.9) vs. collapse severity\nCompare true label noise vs. synthetic model collapse', ha='center', va='center')

    ax.arrow(0.75, 0.4, 0, -0.05, head_width=0.02, head_length=0.02, fc='k', ec='k')

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf")
    plt.close()

if __name__ == "__main__":
    draw_setup_figure("figures/fig1_setup")
    print("Generated figures/fig1_setup.png/pdf")
