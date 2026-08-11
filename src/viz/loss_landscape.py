import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_loss_landscape(history_pure, history_collapse, output_dir: Path):
    """Generate a conceptual plot of loss landscape geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_p = [e["step"] for e in history_pure]
    loss_p = [e["train_loss"] for e in history_pure]

    steps_c = [e["step"] for e in history_collapse]
    loss_c = [e["train_loss"] for e in history_collapse]

    plt.figure(figsize=(10, 6))
    plt.plot(steps_p, loss_p, label="Pure (Sharp Grokking Cliff)", color="green", linewidth=2)
    plt.plot(steps_c, loss_c, label="Severe Collapse (Flatter landscape)", color="red", linestyle="--", linewidth=2)

    plt.yscale("log")
    plt.title("Loss Landscape Geometry via Train Loss Trajectories")
    plt.xlabel("Training Step")
    plt.ylabel("Log Train Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out = output_dir / "loss_landscape_geometry"
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches='tight')
    plt.savefig(out.with_suffix(".pdf"), dpi=150, bbox_inches='tight')
    plt.close()
