import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys
import os

# Add parent dir to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rigorous_stats.stats_module import fit_power_law, fit_log_linear, bootstrap_ci, fit_logistic, bootstrap_logistic_ci

# Constants
GROK_THRESHOLD = 0.99

def parse_scaling_results(grid_dir: Path) -> List[Dict[str, Any]]:
    """Parse results.json from all subdirectories into a flat list of dicts."""
    rows = []
    if not grid_dir.exists():
        return rows

    for path in grid_dir.rglob("results.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            config = data.get("config", {})
            width = config.get("d_model", 128)
            # Default width to 128 if not found
            if width is None:
                width = 128

            # Default dataset size
            dataset_size = config.get("dataset_size", 1.0)
            if dataset_size is None:
                dataset_size = 1.0

            # Look for fraction of synthetic data or 'severity'
            fraction = config.get("collapse_level", 0.0)
            if fraction is None:
                fraction = 0.0

            seed = config.get("seed", 0)
            test_acc = data.get("final_test_acc", 0.0)
            grokking_step = data.get("grokking_step")
            if grokking_step is None:
                grokking_step = -1

            rows.append({
                "width": float(width),
                "dataset_size": float(dataset_size),
                "fraction": float(fraction),
                "seed": int(seed),
                "final_test_acc": test_acc,
                "grokking_step": grokking_step
            })
        except Exception as e:
            print(f"Warning: Failed to parse {path}: {e}")

    return rows

def aggregate_boundary(rows: List[Dict[str, Any]]) -> Dict[Tuple[float, float, int], float]:
    """
    Finds the fraction threshold per (width, seed) where the model stops grokking.
    Similar to threshold_theory.py, returns the *smallest fraction where it failed to grok*.
    """
    by_key: Dict[Tuple[float, int], List[Tuple[float, bool]]] = {}
    for r in rows:
        by_key.setdefault((r["width"], r["dataset_size"], r["seed"]), []).append(
            (r["fraction"], r["final_test_acc"] >= GROK_THRESHOLD)
        )

    out: Dict[Tuple[float, float, int], float] = {}
    for k, lst in by_key.items():
        lst.sort(key=lambda t: t[0])
        fractions = [f for f, _ in lst]
        groks = [g for _, g in lst]
        max_frac = max(fractions) if fractions else 0.0

        if not any(groks):
            out[k] = 0.0
            continue

        first_fail = None
        for f, g in lst:
            if not g:
                first_fail = f
                break

        if first_fail is None:
            out[k] = max_frac + 0.05
        else:
            out[k] = first_fail

    return out
def generate_phase_diagram(rows: List[Dict[str, Any]], out_dir: Path):
    """
    Generate phase diagram (synthetic fraction x model width) colored by grokking outcome.
    Saves PNG and CSV.
    """
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = out_dir / "phase_diagram.csv"
    with open(csv_path, "w") as f:
        f.write("width,fraction,seed,test_acc,grokking_step,outcome\n")
        for r in rows:
            grok = r["final_test_acc"] >= GROK_THRESHOLD
            delayed = grok and r["grokking_step"] > 10000
            if not grok:
                outcome = "no-grok"
            elif delayed:
                outcome = "delayed-grok"
            else:
                outcome = "grok"

            f.write(f"{r['width']},{r['fraction']},{r['seed']},{r['final_test_acc']},{r['grokking_step']},{outcome}\n")

    # Generate Plot
    if not HAS_MATPLOTLIB:
        print("Matplotlib not found, skipping PNG generation.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Aggregate outcomes by taking median behavior across seeds for each (width, fraction)
    by_cell = {}
    for r in rows:
        key = (r["width"], r["fraction"])
        if key not in by_cell:
            by_cell[key] = []
        by_cell[key].append(r)

    xs, ys, colors = [], [], []
    for (width, fraction), cell_rows in by_cell.items():
        accs = [cr["final_test_acc"] for cr in cell_rows]
        steps = [cr["grokking_step"] for cr in cell_rows]

        med_acc = np.median(accs)
        med_step = np.median([s for s in steps if s > 0]) if any(s > 0 for s in steps) else -1

        grok = med_acc >= GROK_THRESHOLD
        delayed = grok and med_step > 10000

        xs.append(width)
        ys.append(fraction)

        if not grok:
            colors.append("red") # no-grok
        elif delayed:
            colors.append("orange") # delayed-grok
        else:
            colors.append("green") # grok

    scatter = ax.scatter(xs, ys, c=colors, s=100, alpha=0.7, edgecolors="k")

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Grok'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Delayed Grok'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='No Grok')
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_xscale("log")
    ax.set_xlabel("Model Width (d_model)")
    ax.set_ylabel("Synthetic Data Fraction")
    ax.set_title("Grokking Phase Diagram")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "phase_diagram.png", dpi=150)
    plt.close(fig)

def fit_scaling_law(per_seed: Dict[Tuple[float, float, int], float], rows: List[Dict[str, Any]], out_dir: Path):
    """
    Fits log-linear and power-law scaling for grokking onset vs model width and dataset size.
    Fits logistic regression for failure-to-grok probability vs fraction and width/size.
    Outputs markdown summary.
    """
    summary_path = out_dir / "scaling_summary.md"

    # 1. Fits for collapse threshold η* vs Model Width
    # Use baseline dataset_size (e.g. 1.0)
    width_pairs = [(w, e) for (w, d, _), e in per_seed.items() if e > 0 and d == 1.0]

    # 2. Fits for collapse threshold η* vs Dataset Size
    # Use baseline width (e.g. 128.0)
    ds_pairs = [(d, e) for (w, d, _), e in per_seed.items() if e > 0 and w == 128.0]

    # 3. Fits for grokking onset step (time to grok) vs synthetic fraction (only for runs that grokked)
    grok_rows = [r for r in rows if r["final_test_acc"] >= GROK_THRESHOLD and r["grokking_step"] > 0]

    # 4. Logistic fit for failure-to-grok vs fraction
    frac_arr = np.array([r["fraction"] for r in rows])
    fail_arr = np.array([1 if r["final_test_acc"] < GROK_THRESHOLD else 0 for r in rows])

    with open(summary_path, "w") as f:
        f.write("# Grokking Collapse Boundary Scaling\n\n")

        # Width scaling
        if width_pairs:
            w_arr = np.array([p[0] for p in width_pairs])
            e_arr = np.array([p[1] for p in width_pairs])
            pl_fit = fit_power_law(w_arr, e_arr)
            ll_fit = fit_log_linear(w_arr, e_arr)

            f.write("## Scaling with Model Width (d_model)\n")
            f.write(f"Power Law (η* = C * d_model^b): C={pl_fit[0]:.4f}, b={pl_fit[1]:.4f}, R2={pl_fit[2]:.4f}\n")
            f.write(f"Log-Linear (log η* = a + b * d_model): a={ll_fit[0]:.4f}, b={ll_fit[1]:.4f}, R2={ll_fit[2]:.4f}\n\n")

        # Dataset size scaling
        if ds_pairs:
            d_arr = np.array([p[0] for p in ds_pairs])
            e_arr = np.array([p[1] for p in ds_pairs])
            pl_fit = fit_power_law(d_arr, e_arr)
            ll_fit = fit_log_linear(d_arr, e_arr)

            f.write("## Scaling with Dataset Size\n")
            f.write(f"Power Law (η* = C * size^b): C={pl_fit[0]:.4f}, b={pl_fit[1]:.4f}, R2={pl_fit[2]:.4f}\n")
            f.write(f"Log-Linear (log η* = a + b * size): a={ll_fit[0]:.4f}, b={ll_fit[1]:.4f}, R2={ll_fit[2]:.4f}\n\n")

        # Time to grok
        if grok_rows:
            g_frac = np.array([r["fraction"] for r in grok_rows])
            g_step = np.array([r["grokking_step"] for r in grok_rows])
            pl_fit = fit_power_law(g_frac, g_step)
            ll_fit = fit_log_linear(g_frac, g_step)

            f.write("## Grokking Onset Step (Time to Grok) vs Synthetic Fraction\n")
            f.write(f"Power Law (step = C * frac^b): C={pl_fit[0]:.4f}, b={pl_fit[1]:.4f}, R2={pl_fit[2]:.4f}\n")
            f.write(f"Log-Linear (log step = a + b * frac): a={ll_fit[0]:.4f}, b={ll_fit[1]:.4f}, R2={ll_fit[2]:.4f}\n\n")

        # Logistic
        if len(frac_arr) > 0:
            log_fit = fit_logistic(frac_arr, fail_arr)
            f.write("## Failure-to-Grok Indicator vs Synthetic Fraction\n")
            f.write(f"Logistic Regression (logit(p) = a + b * frac): a={log_fit[0]:.4f}, b={log_fit[1]:.4f}, pseudo-R2={log_fit[2]:.4f}\n")
            log_ci = bootstrap_logistic_ci(frac_arr, fail_arr)
            f.write(f"- 95% CI for slope b: [{log_ci[0]:.4f}, {log_ci[1]:.4f}]\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/grid", help="Path to sweep results")
    parser.add_argument("--out-dir", type=str, default="analysis/scaling", help="Output directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing results from {results_dir}...")
    rows = parse_scaling_results(results_dir)
    print(f"Found {len(rows)} result rows.")

    if not rows:
        print("No results found. Exiting.")
        return

    generate_phase_diagram(rows, out_dir)
    print(f"Generated phase diagram at {out_dir}")

    boundary = aggregate_boundary(rows)
    fit_scaling_law(boundary, rows, out_dir)
    print(f"Generated scaling fits summary at {out_dir}")

if __name__ == "__main__":
    main()

# Added type hints for function arguments and return types.
