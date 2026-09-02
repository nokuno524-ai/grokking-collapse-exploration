import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.grok_detector import detect_grok_step, is_never_grok, compute_grok_ci

def parse_sweep_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Parse all results.json files in the sweep directory."""
    results = []

    # Path layout: d{d_model}/f{train_fraction}/c{collapse_level}/seed_{seed}/results.json
    for results_file in sweep_dir.rglob("results.json"):
        try:
            with open(results_file, "r") as f:
                data = json.load(f)

            config = data.get("config", {})
            history = data.get("history", [])

            if not history:
                continue

            d_model = config.get("d_model", 128)
            train_fraction = config.get("train_fraction", 0.3)
            collapse_level = config.get("collapse_level", 0.0)
            seed = config.get("seed", 42)

            grok_step = detect_grok_step(history)
            never_grok = is_never_grok(history)

            ci_lower, ci_upper = compute_grok_ci(history)

            results.append({
                "d_model": d_model,
                "train_fraction": train_fraction,
                "collapse_level": collapse_level,
                "seed": seed,
                "grok_step": grok_step,
                "never_grok": never_grok,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })
        except Exception as e:
            print(f"Error parsing {results_file}: {e}")

    return results

def aggregate_by_cell(results: List[Dict[str, Any]]) -> Dict[int, Dict]:
    """
    Aggregate results by d_model, then by (train_fraction, collapse_level).
    Averages grok step across seeds.
    """
    # Group by d_model
    by_d_model = {}
    for r in results:
        d = r["d_model"]
        if d not in by_d_model:
            by_d_model[d] = {}

        f = r["train_fraction"]
        c = r["collapse_level"]
        key = (f, c)

        if key not in by_d_model[d]:
            by_d_model[d][key] = []

        by_d_model[d][key].append(r)

    aggregated = {}
    for d, cell_data in by_d_model.items():
        aggregated[d] = {}
        for (f, c), runs in cell_data.items():
            grok_steps = [r["grok_step"] for r in runs if r["grok_step"] is not None]
            never_groks = sum(1 for r in runs if r["never_grok"])

            mean_grok = np.mean(grok_steps) if grok_steps else None
            std_grok = np.std(grok_steps) if len(grok_steps) > 1 else None

            aggregated[d][(f, c)] = {
                "mean_grok_step": mean_grok,
                "std_grok_step": std_grok,
                "never_grok_count": never_groks,
                "total_seeds": len(runs),
                "runs": runs
            }

    return aggregated

def plot_phase_diagram(aggregated: Dict[int, Dict], output_dir: Path):
    """
    Plot heatmaps for each d_model.
    X-axis: collapse_level (synthetic fraction)
    Y-axis: train_fraction
    Color: grok step (or gray if never groks).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for d_model, cells in aggregated.items():
        # Get unique f and c values
        f_vals = sorted(list(set(k[0] for k in cells.keys())))
        c_vals = sorted(list(set(k[1] for k in cells.keys())))

        # Create matrix
        grok_matrix = np.full((len(f_vals), len(c_vals)), np.nan)
        never_grok_matrix = np.zeros((len(f_vals), len(c_vals)), dtype=bool)

        for i, f in enumerate(f_vals):
            for j, c in enumerate(c_vals):
                cell = cells.get((f, c))
                if cell:
                    # If more than half the seeds never grokked, mark as never grok
                    if cell["never_grok_count"] > cell["total_seeds"] / 2:
                        never_grok_matrix[i, j] = True
                    elif cell["mean_grok_step"] is not None:
                        grok_matrix[i, j] = cell["mean_grok_step"]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap of grok steps
        im = ax.imshow(grok_matrix, cmap='viridis_r', origin='lower', aspect='auto')

        # Overlay never grok cells with gray or hatched pattern
        # Since NaN values are white by default, we can just let them be, or specifically color them.

        # Add text annotations
        for i, f in enumerate(f_vals):
            for j, c in enumerate(c_vals):
                if never_grok_matrix[i, j]:
                    ax.text(j, i, 'Never\nGrok', ha='center', va='center', color='black')
                elif not np.isnan(grok_matrix[i, j]):
                    ax.text(j, i, f'{int(grok_matrix[i, j])}', ha='center', va='center', color='white')

        ax.set_xticks(np.arange(len(c_vals)))
        ax.set_yticks(np.arange(len(f_vals)))
        ax.set_xticklabels([f"{c:.2f}" for c in c_vals])
        ax.set_yticklabels([f"{f:.2f}" for f in f_vals])

        ax.set_xlabel('Synthetic Data Fraction (collapse_level)')
        ax.set_ylabel('Dataset Size Multiplier (train_fraction)')
        ax.set_title(f'Phase Diagram: Grokking Step (d_model={d_model})')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Grokking Step')

        plt.tight_layout()
        out_path = output_dir / f"phase_diagram_d{d_model}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

def generate_markdown_report(aggregated: Dict[int, Dict], output_path: Path):
    with open(output_path, "w") as f:
        f.write("# Grokking Phase Diagram Scaling Characterization\n\n")

        for d_model, cells in sorted(aggregated.items()):
            f.write(f"## Model Size: d_model = {d_model}\n\n")
            f.write(f"![Phase Diagram](phase_diagram_d{d_model}.png)\n\n")

            f.write("| Train Fraction | Synthetic Fraction | Mean Grok Step | Never Grok Count | Total Seeds |\n")
            f.write("|---|---|---|---|---|\n")

            for (frac, synth), cell in sorted(cells.items()):
                grok = f"{cell['mean_grok_step']:.1f}" if cell['mean_grok_step'] is not None else "N/A"
                f.write(f"| {frac:.2f} | {synth:.2f} | {grok} | {cell['never_grok_count']} | {cell['total_seeds']} |\n")
            f.write("\n")
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/phase_diagram")
    parser.add_argument("--output-dir", type=str, default="results/phase_diagram")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    results = parse_sweep_results(results_dir)
    if not results:
        print("No results found.")
        return

    aggregated = aggregate_by_cell(results)
    plot_phase_diagram(aggregated, out_dir)
    generate_markdown_report(aggregated, out_dir / "grokking_phase_diagram.md")

if __name__ == "__main__":
    main()
