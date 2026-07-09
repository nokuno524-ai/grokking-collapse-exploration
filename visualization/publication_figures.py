import os
import matplotlib.pyplot as plt
import numpy as np

# Import functions from other visualization modules
from visualization.attention_evolution import (
    plot_attention_heatmaps,
    animate_attention_evolution
)
from visualization.training_dynamics import (
    plot_training_dynamics,
    plot_fourier_evolution,
    plot_condition_overlay
)
from visualization.collapse_landscape import (
    plot_3d_accuracy_surface,
    plot_grokking_onset_heatmap,
    plot_weight_norm_trajectory
)

def set_publication_style():
    """
    Set matplotlib parameters for publication-quality figures
    (NeurIPS/ICML style: high DPI, correct fonts, colorblind friendly).
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2.0,
        # Colorblind friendly cycle (Okabe-Ito)
        'axes.prop_cycle': plt.cycler('color', [
            '#E69F00', '#56B4E9', '#009E73',
            '#F0E442', '#0072B2', '#D55E00',
            '#CC79A7', '#000000'
        ])
    })

def generate_all_figures(output_dir: str = 'analysis/figures'):
    """
    Generate mock publication figures using the visualization suite.
    In a real scenario, this would load actual experiment data.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Example 1: Training dynamics
    # Generate some mock history
    steps = np.arange(0, 5000, 100)
    history = []
    for s in steps:
        h = {
            'step': int(s),
            'train_loss': np.exp(-s/500) + 0.1,
            'test_loss': np.exp(-s/1000) + 0.2 if s < 2500 else np.exp(-(s-2500)/100) + 0.05,
            'train_acc': 1 - np.exp(-s/500),
            'test_acc': 0.1 if s < 2500 else 1 - np.exp(-(s-2500)/100),
            'weight_norm': 10 + 5 * np.sin(s/1000),
            'fourier_concentration': 0.1 if s < 2500 else 0.8 * (1 - np.exp(-(s-2500)/200)),
        }
        history.append(h)

    plot_training_dynamics(history, os.path.join(output_dir, 'fig1_training_dynamics.pdf'))
    plot_training_dynamics(history, os.path.join(output_dir, 'fig1_training_dynamics.png'))

    # Example 2: 3D Surface
    levels = np.linspace(0, 1, 10)
    stps = np.arange(0, 5000, 500)
    # Mock accuracies: drops as level increases, rises as step increases
    L, S = np.meshgrid(levels, stps)
    acc = (1 - L) * (1 - np.exp(-S/1000))
    # Transpose back to match levels, steps indexing
    acc = acc.T

    plot_3d_accuracy_surface(levels, stps, acc, os.path.join(output_dir, 'fig2_collapse_surface.pdf'))
    plot_3d_accuracy_surface(levels, stps, acc, os.path.join(output_dir, 'fig2_collapse_surface.png'))

    # Example 3: Weight norm trajectory with multiple conditions
    histories = {
        'Pure (0.0)': history,
        'Low (0.3)': [{'step': s, 'weight_norm': 12 + 4 * np.sin(s/1000)} for s in steps],
        'Severe (0.9)': [{'step': s, 'weight_norm': 15 + 2 * np.sin(s/1000)} for s in steps],
    }

    # Mock CIs
    cis = {}
    for k, hist in histories.items():
        norms = np.array([h['weight_norm'] for h in hist])
        cis[k] = (norms - 1.0, norms + 1.0)

    plot_weight_norm_trajectory(histories, os.path.join(output_dir, 'fig3_weight_norms.pdf'), confidence_intervals=cis)
    plot_weight_norm_trajectory(histories, os.path.join(output_dir, 'fig3_weight_norms.png'), confidence_intervals=cis)

    print(f"Generated publication figures in {output_dir}/")

if __name__ == '__main__':
    generate_all_figures()
