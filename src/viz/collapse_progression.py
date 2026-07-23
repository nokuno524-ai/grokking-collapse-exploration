import plotly.graph_objects as go
import json
from pathlib import Path

def plot_collapse_progression(results_dir: str):
    """
    Plots the final test accuracy and grokking step across collapse levels.
    Requires a specific directory structure with 'pure', 'low_collapse', etc.
    """
    base_path = Path(results_dir)
    levels = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    accs = []
    grok_steps = []
    valid_levels = []

    for level in levels:
        res_file = base_path / level / "results.json"
        if res_file.exists():
            with open(res_file, 'r') as f:
                data = json.load(f)
                accs.append(data.get("final_test_acc", 0))
                grok_steps.append(data.get("grokking_step") or 0)
                valid_levels.append(level)

    if not valid_levels:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=valid_levels, y=accs, name="Final Test Accuracy", yaxis="y1"))

    fig.add_trace(go.Scatter(x=valid_levels, y=grok_steps, name="Grokking Step", yaxis="y2", mode="lines+markers"))

    fig.update_layout(
        title="Collapse Progression",
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        yaxis2=dict(title="Grokking Step", overlaying="y", side="right"),
        barmode="group"
    )
    return fig
