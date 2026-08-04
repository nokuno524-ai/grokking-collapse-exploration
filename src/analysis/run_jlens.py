import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.model import ModularArithmeticTransformer
from src.analysis.jlens import JLensAnalyzer

CONDITIONS = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

def run_analysis(results_dir="results"):
    results_path = Path(results_dir)

    # Store metrics: condition -> layer -> step -> [rank, entropy]
    metrics_history = {cond: {'embedding': {'steps': [], 'rank': [], 'entropy': []},
                              'transformer': {'steps': [], 'rank': [], 'entropy': []},
                              'layer_norm': {'steps': [], 'rank': [], 'entropy': []}}
                       for cond in CONDITIONS}

    # Generate test inputs
    x_test = torch.randint(0, 59, (100, 2))

    for condition in CONDITIONS:
        cond_dir = results_path / condition
        if not cond_dir.exists():
            print(f"Directory {cond_dir} not found. Skipping.")
            continue

        checkpoints = glob.glob(str(cond_dir / "checkpoint_*.pt"))
        if not checkpoints:
            print(f"No checkpoints found in {cond_dir}. Skipping.")
            continue

        def extract_step(ckpt_path):
            filename = os.path.basename(ckpt_path)
            step_str = filename.split('_')[1].split('.')[0]
            return int(step_str)

        checkpoints.sort(key=extract_step)

        for ckpt in checkpoints:
            step = extract_step(ckpt)

            # Load model
            model = ModularArithmeticTransformer()
            try:
                state_dict = torch.load(ckpt, map_location='cpu')
                # Try to extract the model state_dict if it's nested
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'model_state' in state_dict:
                    state_dict = state_dict['model_state']

                # Check for module. prefix in state dict
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"Failed to load {ckpt}: {e}")
                continue

            analyzer = JLensAnalyzer(model)
            metrics = analyzer.analyze(x_test)

            for layer_name in metrics:
                metrics_history[condition][layer_name]['steps'].append(step)
                metrics_history[condition][layer_name]['rank'].append(metrics[layer_name]['rank'])
                metrics_history[condition][layer_name]['entropy'].append(metrics[layer_name]['entropy'])

    # Plotting
    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    layers = ['embedding', 'transformer', 'layer_norm']

    fig, axes = plt.subplots(len(layers), 2, figsize=(15, 12))

    for i, layer in enumerate(layers):
        for condition in CONDITIONS:
            data = metrics_history[condition][layer]
            if not data['steps']: continue

            axes[i, 0].plot(data['steps'], data['rank'], label=condition, color=colors[condition], alpha=0.8, linewidth=2)
            axes[i, 1].plot(data['steps'], data['entropy'], label=condition, color=colors[condition], alpha=0.8, linewidth=2)

        axes[i, 0].set_title(f"J-Space Rank: {layer}")
        axes[i, 0].set_xlabel("Step")
        axes[i, 0].set_ylabel("Rank")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend()

        axes[i, 1].set_title(f"J-Space Entropy: {layer}")
        axes[i, 1].set_xlabel("Step")
        axes[i, 1].set_ylabel("Entropy")
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig(results_path / "jlens_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved J-Lens analysis plot to results/jlens_analysis.png")

if __name__ == "__main__":
    run_analysis()
