import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

def plot_accuracy_curves(histories: Dict[str, List[Dict]], output_dir: Path):
    """Plot accuracy curves with grokking point annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']

    for i, (condition, history) in enumerate(histories.items()):
        steps = [e["step"] for e in history]
        test_acc = [e.get("test_acc", 0) for e in history]

        c = colors[i % len(colors)]
        plt.plot(steps, test_acc, label=condition, color=c, linewidth=2)

        # Annotate grokking point
        for step, acc in zip(steps, test_acc):
            if acc >= 0.95:
                plt.scatter(step, acc, color=c, s=100, zorder=5, edgecolor='black')
                plt.annotate(f"{condition} Groks", (step, acc), xytext=(10, -15), textcoords='offset points')
                break

    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label="Grokking Threshold")
    plt.title("Accuracy Curves with Grokking Transition Points")
    plt.xlabel("Training Step")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out = output_dir / "accuracy_curves"
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches='tight')
    plt.savefig(out.with_suffix(".pdf"), dpi=150, bbox_inches='tight')
    plt.close()
