#!/usr/bin/env python3
"""
Dashboard script to summarize and visualize experiment results.
Creates an HTML dashboard with embedded matplotlib figures.
"""
from src.management.results import ResultsCollector
import matplotlib.pyplot as plt
import os
import io
import base64
import subprocess
import matplotlib
from pathlib import Path

# Important: Use Agg backend to avoid headless environment crashes
matplotlib.use('Agg')


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Grokking-Collapse Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .status-box {{ display: inline-block; padding: 10px 20px; margin: 5px; border-radius: 5px; font-weight: bold; }}
        .status-active {{ background-color: #e3f2fd; color: #1565c0; }}
        .status-done {{ background-color: #e8f5e9; color: #2e7d32; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #eee; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Grokking-Collapse Results Dashboard</h1>

        <div class="card">
            <h2>Slurm Status</h2>
            {slurm_status}
        </div>

        <div class="card">
            <h2>Overview</h2>
            <p>Found <strong>{num_runs}</strong> completed runs.</p>
            {overview_table}
        </div>

        <div class="card">
            <h2>Accuracy vs Collapse Level</h2>
            <img src="data:image/png;base64,{plot_accuracy}" alt="Accuracy Plot" />
        </div>

        <div class="card">
            <h2>Weight Norm Evolution</h2>
            <img src="data:image/png;base64,{plot_weight}" alt="Weight Norm Plot" />
        </div>

        <div class="card">
            <h2>Phase Transition Timing</h2>
            <img src="data:image/png;base64,{plot_timing}" alt="Timing Plot" />
        </div>
    </div>
</body>
</html>
"""


def get_slurm_status():
    """Query squeue to get active/failed job statuses."""
    try:
        result = subprocess.run(["squeue", "--me"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:
                return "<p>No active SLURM jobs.</p>"

            html = "<table><tr><th>JOBID</th><th>PARTITION</th><th>NAME</th><th>USER</th><th>ST</th><th>TIME</th><th>NODES</th></tr>"
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 7:
                    html += f"<tr><td>{parts[0]}</td><td>{parts[1]}</td><td>{parts[2]}</td><td>{parts[3]}</td><td>{parts[4]}</td><td>{parts[5]}</td><td>{parts[6]}</td></tr>"
            html += "</table>"
            return html
    except Exception:
        pass
    return "<p><em>Slurm queue not available.</em></p>"


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def generate_dashboard(results_dir="results", output_html="dashboard.html"):
    collector = ResultsCollector(results_dir)
    df = collector.aggregate_to_dataframe()

    if df.empty:
        print(f"No results found in {results_dir}.")
        return

    print(f"Aggregated {len(df)} runs.")

    # 1. Overview Table
    cols_to_show = []
    if 'experiment_name' in df.columns:
        cols_to_show.append('experiment_name')
    if 'dataset.collapse_level' in df.columns:
        cols_to_show.append('dataset.collapse_level')
    if 'training.weight_decay' in df.columns:
        cols_to_show.append('training.weight_decay')
    if 'dataset.train_fraction' in df.columns:
        cols_to_show.append('dataset.train_fraction')
    cols_to_show.extend(['grokked', 'grokking_step', 'final_test_acc'])

    # Filter available columns
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    table_html = df[cols_to_show].to_html(index=False, classes="dataframe", border=0)

    # 2. Accuracy Plot
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    if 'dataset.collapse_level' in df.columns and 'final_test_acc' in df.columns:
        # Group by collapse level if there are multiple seeds
        grouped = df.groupby('dataset.collapse_level')['final_test_acc'].agg(['mean', 'std']).reset_index()
        if not grouped.empty:
            ax_acc.errorbar(grouped['dataset.collapse_level'], grouped['mean'],
                            yerr=grouped['std'], marker='o', capsize=5, linestyle='-',
                            linewidth=2, markersize=8)
            ax_acc.set_xlabel('Collapse Level')
            ax_acc.set_ylabel('Final Test Accuracy')
            ax_acc.set_title('Test Accuracy vs Collapse Level')
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_ylim(0, 1.05)
    else:
        ax_acc.text(0.5, 0.5, 'Insufficient data for accuracy plot', ha='center')

    b64_acc = fig_to_base64(fig_acc)

    # 3. Weight Norm Evolution Plot (aggregated across available runs)
    fig_weight, ax_weight = plt.subplots(figsize=(10, 6))
    plotted_weights = False

    # Try to plot a sample of weight norms
    for idx, row in df.iterrows():
        if 'path' in row and os.path.exists(row['path']):
            history_df = collector.load_history(Path(row['path']))
            if not history_df.empty and 'weight_norm' in history_df.columns and 'step' in history_df.columns:
                label = f"Lvl: {row.get('dataset.collapse_level', 'N/A')}, WD: {row.get('training.weight_decay', 'N/A')}"
                # Just plot a subset to avoid clutter (max 10 lines)
                if idx < 10:
                    ax_weight.plot(history_df['step'], history_df['weight_norm'], label=label, alpha=0.7)
                    plotted_weights = True

    if plotted_weights:
        ax_weight.set_xlabel('Step')
        ax_weight.set_ylabel('Weight Norm ‖W‖')
        ax_weight.set_title('Weight Norm Evolution')
        ax_weight.grid(True, alpha=0.3)
        ax_weight.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax_weight.text(0.5, 0.5, 'Insufficient history data for weight norm plot', ha='center')

    b64_weight = fig_to_base64(fig_weight)

    # 4. Timing Plot
    fig_time, ax_time = plt.subplots(figsize=(10, 6))
    if 'dataset.collapse_level' in df.columns and 'grokking_step' in df.columns:
        # Only plot where grokked is True
        grokked_df = df[df['grokked'] is True]
        if not grokked_df.empty:
            grouped = grokked_df.groupby('dataset.collapse_level')['grokking_step'].agg(['mean', 'std']).reset_index()
            ax_time.errorbar(grouped['dataset.collapse_level'], grouped['mean'],
                             yerr=grouped['std'], marker='s', capsize=5, linestyle='--',
                             linewidth=2, markersize=8, color='orange')
            ax_time.set_xlabel('Collapse Level')
            ax_time.set_ylabel('Grokking Step')
            ax_time.set_title('Grokking Step vs Collapse Level')
            ax_time.grid(True, alpha=0.3)
        else:
            ax_time.text(0.5, 0.5, 'No runs have grokked yet', ha='center')
    else:
        ax_time.text(0.5, 0.5, 'Insufficient data for timing plot', ha='center')

    b64_time = fig_to_base64(fig_time)

    # Generate HTML
    html = HTML_TEMPLATE.format(
        slurm_status=get_slurm_status(),
        num_runs=len(df),
        overview_table=table_html,
        plot_accuracy=b64_acc,
        plot_weight=b64_weight,
        plot_timing=b64_time
    )

    with open(output_html, 'w') as f:
        f.write(html)

    print(f"Dashboard generated successfully at {output_html}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate experiment dashboard")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Output HTML file path")
    args = parser.parse_args()

    generate_dashboard(args.results_dir, args.output)
