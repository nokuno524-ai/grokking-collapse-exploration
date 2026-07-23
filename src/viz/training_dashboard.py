import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import List, Dict

def plot_training_dashboard(results_path: str):
    """
    Creates a dashboard with loss, accuracy, weight norm, and fourier concentration.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        raise ValueError(f"No history found in {results_path}")

    steps = [e["step"] for e in history]

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss", "Accuracy", "Weight Norm", "Attention Entropy"))

    # Loss
    fig.add_trace(go.Scatter(x=steps, y=[e["train_loss"] for e in history], name="Train Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=[e["test_loss"] for e in history], name="Test Loss"), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(x=steps, y=[e["train_acc"] for e in history], name="Train Acc"), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=[e["test_acc"] for e in history], name="Test Acc"), row=1, col=2)

    # Weight Norm
    fig.add_trace(go.Scatter(x=steps, y=[e["weight_norm"] for e in history], name="Weight Norm"), row=2, col=1)

    # Attention Entropy
    fig.add_trace(go.Scatter(x=steps, y=[e["attention_entropy"] for e in history], name="Fourier Conc"), row=2, col=2)

    fig.update_layout(height=800, title_text="Training Dynamics Dashboard")
    return fig
