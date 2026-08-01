import argparse
import os
import json
import numpy as np
from pathlib import Path

# Try importing dependencies cleanly
try:
    from src.train import TrainConfig, train
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.train import TrainConfig, train

def generate_phase_diagram_data(output_root: str, max_steps: int):
    """
    Run a grid search over collapse severities to generate phase diagram data.
    X-axis: Collapse Severity
    Y-axis: Training Step (logged in history)
    Color: Accuracy (logged in history)
    """
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for severity in severities:
        print(f"Running phase diagram sweep for severity {severity}...")

        # When severity is 0, we treat it as pure (collapse_level=0)
        # otherwise we fix collapse level and sweep severity.
        level = 0.0 if severity == 0.0 else 0.5

        train_config = TrainConfig(
            prime=59,
            train_fraction=0.3,
            lr=1e-3,
            weight_decay=1.0,
            collapse_level=level,
            collapse_severity=severity,
            seed=42,
            condition_name=f"severity_{severity}",
            output_dir=str(Path(output_root) / "phase_diagram"),
            max_steps=max_steps,
            eval_every=100, # More frequent eval for better resolution
            log_every=100,
        )

        state = train(train_config)
        results[severity] = state.history

    return results

def plot_phase_diagram(results, output_dir):
    import matplotlib.pyplot as plt
    import os

    severities = sorted(list(results.keys()))
    steps = [entry['step'] for entry in results[severities[0]]]

    # Create grid for pcolormesh
    # X: severities, Y: steps, Z: test_acc
    X, Y = np.meshgrid(severities, steps)
    Z = np.zeros_like(X, dtype=float)

    for i, step in enumerate(steps):
        for j, severity in enumerate(severities):
            # Find the entry for this step
            history = results[severity]
            acc = 0.0
            for entry in history:
                if entry['step'] == step:
                    acc = entry['test_acc']
                    break
            Z[i, j] = acc

    plt.figure(figsize=(10, 8))
    # Note: pcolormesh needs edges, but we can plot points with scatter or just use imshow
    plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Test Accuracy')

    # Add a contour line for grokking threshold (0.90)
    plt.contour(X, Y, Z, levels=[0.90], colors='red', linestyles='dashed')

    plt.title('Phase Diagram: Grokking vs Collapse Severity')
    plt.xlabel('Collapse Severity')
    plt.ylabel('Training Step')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'phase_diagram.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved phase diagram plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/phase_diagram_data")
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    results = generate_phase_diagram_data(args.output_dir, args.max_steps)
    plot_phase_diagram(results, args.output_dir)
    print(f"Phase diagram generation complete. Outputs saved to {args.output_dir}")
