import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path so we can import scaling_analysis
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from scaling_analysis import run_grokking_sweep, compute_grokking_threshold

def generate_phase_diagram(results, output_path="results/phase_diagram.png"):
    """
    Generates a 2D phase diagram.
    X-axis: Collapse Severity
    Y-axis: Model Size
    Color: Grokking Step (-1 means failure to grok, represented as a distinct color/marker)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sizes = []
    severities = []
    grok_steps = []

    for val in results.values():
        sizes.append(val["model_size"])
        severities.append(val["collapse_severity"])
        grok_steps.append(val["grokking_step"])

    sizes = np.array(sizes)
    severities = np.array(severities)
    grok_steps = np.array(grok_steps)

    # Create grid for plotting
    unique_sizes = sorted(list(set(sizes)))
    unique_sevs = sorted(list(set(severities)))

    Z = np.zeros((len(unique_sizes), len(unique_sevs)))

    for i, s in enumerate(unique_sizes):
        for j, v in enumerate(unique_sevs):
            # Find matching result
            idx = np.where((sizes == s) & (severities == v))[0]
            if len(idx) > 0:
                Z[i, j] = grok_steps[idx[0]]
            else:
                Z[i, j] = np.nan

    # Separate successful grokking from failures for visualization
    Z_success = np.ma.masked_where(Z == -1, Z)

    plt.figure(figsize=(10, 8))

    # Plot successful grokking steps with a colormap
    cmap = plt.cm.viridis_r # reversed so faster (fewer steps) is brighter/different
    im = plt.pcolormesh(unique_sevs, unique_sizes, Z_success, cmap=cmap, shading='nearest')

    # Mark failures with a distinct color (e.g., grey)
    failure_mask = (Z == -1)
    if np.any(failure_mask):
        plt.pcolormesh(unique_sevs, unique_sizes, np.ma.masked_where(~failure_mask, Z),
                      cmap=plt.matplotlib.colors.ListedColormap(['#dddddd']), shading='nearest')

    plt.colorbar(im, label="Grokking Step (Grey = No Grokking)")

    # Plot the threshold boundary
    thresholds = compute_grokking_threshold(results)
    t_sizes = sorted(list(thresholds.keys()))
    t_vals = [thresholds[s] for s in t_sizes]

    plt.plot(t_vals, t_sizes, 'r--', linewidth=2, marker='o', label="Phase Boundary")

    plt.xlabel("Collapse Severity")
    plt.ylabel("Model Size (Params)")
    plt.title("Grokking Phase Diagram")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved phase diagram to {output_path}")

if __name__ == "__main__":
    # Generate mock data
    sizes = [100000, 200000, 400000, 800000, 1600000]
    severities = np.linspace(0.0, 0.6, 15).tolist()

    results = run_grokking_sweep(sizes, severities, dummy_mode=True)
    generate_phase_diagram(results, "results/phase_diagram.png")
