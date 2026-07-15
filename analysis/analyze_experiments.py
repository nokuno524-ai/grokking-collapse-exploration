import json
import os
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def plot_metric_evolution(results_dir, metric_name, ylabel, title, output_prefix):
    plt.figure(figsize=(10, 6))
    fig_plotly = go.Figure()

    conditions = ['pure', 'low_collapse', 'medium_collapse', 'high_collapse', 'severe_collapse']

    for condition in conditions:
        result_file = Path(results_dir) / condition / "results.json"
        if not result_file.exists():
            print(f"Skipping {condition}: results not found.")
            continue

        with open(result_file, "r") as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            continue

        df = pd.DataFrame(history)
        if metric_name not in df.columns:
            continue

        plt.plot(df['step'], df[metric_name], label=condition)
        fig_plotly.add_trace(go.Scatter(x=df['step'], y=df[metric_name], mode='lines', name=condition))

        # Mark grokking step if applicable
        grokking_step = data.get("grokking_step")
        if grokking_step and metric_name in ['test_acc']:
            grok_acc = df.loc[df['step'] == grokking_step, metric_name].values
            if len(grok_acc) > 0:
                plt.scatter([grokking_step], [grok_acc[0]], color='red', zorder=5)
                fig_plotly.add_trace(go.Scatter(x=[grokking_step], y=[grok_acc[0]], mode='markers', marker=dict(color='red', size=10), name=f'{condition} grok'))

    # Matplotlib
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_prefix}.png")
    plt.close()

    # Plotly
    fig_plotly.update_layout(title=title, xaxis_title="Step", yaxis_title=ylabel)
    fig_plotly.write_html(f"{output_prefix}.html")

def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Loss curves
    plot_metric_evolution(args.results_dir, "train_loss", "Loss", "Training Loss Evolution", f"{args.output_dir}/train_loss_evolution")
    plot_metric_evolution(args.results_dir, "test_loss", "Loss", "Test Loss Evolution", f"{args.output_dir}/test_loss_evolution")

    # Accuracy curves
    plot_metric_evolution(args.results_dir, "train_acc", "Accuracy", "Training Accuracy Evolution", f"{args.output_dir}/train_acc_evolution")
    plot_metric_evolution(args.results_dir, "test_acc", "Accuracy", "Test Accuracy Evolution", f"{args.output_dir}/test_acc_evolution")

    # Weight norm evolution
    plot_metric_evolution(args.results_dir, "weight_norm", "L2 Norm", "Weight Norm Evolution", f"{args.output_dir}/weight_norm_evolution")

if __name__ == "__main__":
    main()
