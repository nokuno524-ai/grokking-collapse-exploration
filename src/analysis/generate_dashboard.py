import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from typing import List, Dict

def load_results(results_dir: str) -> Dict[str, dict]:
    """Load results from all conditions."""
    results = {}
    base_path = Path(results_dir)

    if not base_path.exists():
        print(f"Directory {results_dir} not found.")
        return results

    for condition_dir in base_path.iterdir():
        if condition_dir.is_dir():
            result_file = condition_dir / "results.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    results[condition_dir.name] = json.load(f)

    return results

def generate_dashboard(results_dir: str, output_path: str = "dashboard.html"):
    """Generate comprehensive HTML dashboard using Plotly."""
    results = load_results(results_dir)
    if not results:
        print("No results to plot.")
        return

    # Setup subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Test Accuracy", "Train Loss", "Weight Norm", "Fourier Concentration"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    colors = {
        "pure": "blue",
        "low_collapse": "green",
        "medium_collapse": "orange",
        "high_collapse": "red",
        "severe_collapse": "purple",
        "test_run": "black"
    }

    for condition, data in results.items():
        history = data.get("history", [])
        if not history:
            continue

        steps = [entry["step"] for entry in history]
        test_acc = [entry["test_acc"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        weight_norm = [entry["weight_norm"] for entry in history]
        fourier = [entry["fourier_concentration"] for entry in history]

        color = colors.get(condition, "grey")

        # Test Acc
        fig.add_trace(go.Scatter(x=steps, y=test_acc, mode="lines", name=condition, line=dict(color=color), legendgroup=condition), row=1, col=1)
        # Train Loss
        fig.add_trace(go.Scatter(x=steps, y=train_loss, mode="lines", name=condition, line=dict(color=color), legendgroup=condition, showlegend=False), row=1, col=2)
        # Weight Norm
        fig.add_trace(go.Scatter(x=steps, y=weight_norm, mode="lines", name=condition, line=dict(color=color), legendgroup=condition, showlegend=False), row=2, col=1)
        # Fourier
        fig.add_trace(go.Scatter(x=steps, y=fourier, mode="lines", name=condition, line=dict(color=color), legendgroup=condition, showlegend=False), row=2, col=2)

    fig.update_layout(
        title="Grokking and Model Collapse Experiment Dashboard",
        height=800,
        width=1200,
        template="plotly_white"
    )

    # Save to HTML
    fig.write_html(output_path)
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    import sys
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    generate_dashboard(res_dir)
