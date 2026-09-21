import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import sys

sys.path.append('.') # Ensure src can be imported
from src.analysis.atlas_analysis import render_atlas_figure

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Attention & Loss Analysis Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        .comparison-container { display: flex; flex-direction: row; gap: 40px; }
        .condition-column { flex: 1; min-width: 0; }
        .atlas-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .atlas-card { border: 1px solid #ccc; padding: 10px; border-radius: 5px; text-align: center; }
        .metrics-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
        .metric-plot { border: 1px solid #ccc; padding: 10px; border-radius: 5px; }
        .section-divider { border-top: 2px solid #ccc; margin: 40px 0; }
    </style>
</head>
<body>
    <h1>Attention & Loss Evolution Report</h1>

    <h2>Loss Curves (Grokked vs. Collapsed)</h2>
    <div class="metrics-container">
        <div class="metric-plot">
            <img src="data:image/png;base64,{loss_plot}" />
        </div>
    </div>

    <div class="section-divider"></div>

    <div class="comparison-container">
        <!-- Condition 1 Column -->
        <div class="condition-column">
            <h2>Condition: {name1}</h2>

            <h3>Metrics over Training</h3>
            <div class="metrics-container">
                <div class="metric-plot">
                    <h4>Entropy</h4>
                    <img src="data:image/png;base64,{entropy_plot1}" />
                </div>
                <div class="metric-plot">
                    <h4>Diagonal Mass</h4>
                    <img src="data:image/png;base64,{diag_mass_plot1}" />
                </div>
                <div class="metric-plot">
                    <h4>Head Specialization</h4>
                    <img src="data:image/png;base64,{spec_plot1}" />
                </div>
            </div>

            <h3>Attention Atlas by Step</h3>
            <div class="atlas-container">
                {atlas_cards1}
            </div>
        </div>

        <!-- Condition 2 Column -->
        <div class="condition-column">
            <h2>Condition: {name2}</h2>

            <h3>Metrics over Training</h3>
            <div class="metrics-container">
                <div class="metric-plot">
                    <h4>Entropy</h4>
                    <img src="data:image/png;base64,{entropy_plot2}" />
                </div>
                <div class="metric-plot">
                    <h4>Diagonal Mass</h4>
                    <img src="data:image/png;base64,{diag_mass_plot2}" />
                </div>
                <div class="metric-plot">
                    <h4>Head Specialization</h4>
                    <img src="data:image/png;base64,{spec_plot2}" />
                </div>
            </div>

            <h3>Attention Atlas by Step</h3>
            <div class="atlas-container">
                {atlas_cards2}
            </div>
        </div>
    </div>
</body>
</html>
"""

def plot_metric(steps, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(steps, values, marker='o')
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_loss_curves(results1, results2, name1, name2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (res, name) in enumerate([(results1, name1), (results2, name2)]):
        if not res or "history" not in res:
            continue

        history = res["history"]
        steps = [entry["step"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        test_loss = [entry["test_loss"] for entry in history]

        ax = axes[idx]
        ax.plot(steps, train_loss, label='Train Loss')
        ax.plot(steps, test_loss, label='Test Loss')
        ax.set_yscale('log')
        ax.set_title(f"{name} Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def parse_metrics_and_atlas(data_dir):
    data_dir = Path(data_dir)
    steps, entropies, diag_masses, specs = [], [], [], []

    metrics_file = data_dir / "attention_metrics.csv"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                entropies.append(float(row["entropy"]))
                diag_masses.append(float(row["diagonal_mass"]))
                specs.append(float(row["head_specialization"]))

    if steps:
        sorted_idx = np.argsort(steps)
        steps = np.array(steps)[sorted_idx]
        entropies = np.array(entropies)[sorted_idx]
        diag_masses = np.array(diag_masses)[sorted_idx]
        specs = np.array(specs)[sorted_idx]

    entropy_b64 = plot_metric(steps, entropies, "Attention Entropy", "Entropy") if steps.size > 0 else ""
    diag_b64 = plot_metric(steps, diag_masses, "Diagonal Mass", "Fraction") if steps.size > 0 else ""
    spec_b64 = plot_metric(steps, specs, "Head Specialization", "Variance") if steps.size > 0 else ""

    atlas_cards_html = ""
    npz_files = sorted(data_dir.glob("attention_step_*.npz"), key=lambda x: int(x.stem.split("_")[2]))

    for fpath in npz_files:
        step = int(fpath.stem.split("_")[2])
        data = np.load(fpath)
        layer_name = list(data.keys())[0]
        attn = data[layer_name]
        mean_attn = attn.mean(axis=0)
        img_b64 = render_atlas_figure(mean_attn, title=f"Step {step}")

        card = f"""
        <div class="atlas-card">
            <h4>Step {step}</h4>
            <img src="data:image/png;base64,{img_b64}" />
        </div>
        """
        atlas_cards_html += card

    return entropy_b64, diag_b64, spec_b64, atlas_cards_html

def load_results_json(data_dir):
    results_path = Path(data_dir) / "results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from attention data.")
    parser.add_argument("--dir1", type=str, required=True, help="First directory containing results/attention data.")
    parser.add_argument("--name1", type=str, default="Grokked", help="Label for first directory.")
    parser.add_argument("--dir2", type=str, required=True, help="Second directory containing results/attention data.")
    parser.add_argument("--name2", type=str, default="Collapsed", help="Label for second directory.")
    parser.add_argument("--output", type=str, default="attention_report.html", help="Path to output HTML file.")
    args = parser.parse_args()

    # Process Loss Curves
    results1 = load_results_json(args.dir1)
    results2 = load_results_json(args.dir2)
    loss_b64 = plot_loss_curves(results1, results2, args.name1, args.name2)

    # Process Metrics and Atlas
    ent1, diag1, spec1, atlas1 = parse_metrics_and_atlas(args.dir1)
    ent2, diag2, spec2, atlas2 = parse_metrics_and_atlas(args.dir2)

    final_html = HTML_TEMPLATE.replace("{loss_plot}", loss_b64)\
                              .replace("{name1}", args.name1)\
                              .replace("{entropy_plot1}", ent1)\
                              .replace("{diag_mass_plot1}", diag1)\
                              .replace("{spec_plot1}", spec1)\
                              .replace("{atlas_cards1}", atlas1)\
                              .replace("{name2}", args.name2)\
                              .replace("{entropy_plot2}", ent2)\
                              .replace("{diag_mass_plot2}", diag2)\
                              .replace("{spec_plot2}", spec2)\
                              .replace("{atlas_cards2}", atlas2)

    with open(args.output, "w") as f:
        f.write(final_html)

    print(f"Report generated at {args.output}")

if __name__ == "__main__":
    main()
