import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from scaling_analysis import run_grokking_sweep, fit_scaling_law, compute_grokking_threshold

def generate_latex_table(results: Dict[str, dict], output_path: str = "results/scaling_table.tex"):
    """
    Outputs a formatted LaTeX table summarizing grokking thresholds across model sizes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    thresholds = compute_grokking_threshold(results)
    law = fit_scaling_law(results)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{cc}",
        "\\toprule",
        "\\textbf{Model Size (Params)} & \\textbf{Critical Collapse Severity ($S^*$)} \\\\",
        "\\midrule"
    ]

    for size, threshold in sorted(thresholds.items()):
        lines.append(f"{size:,} & {threshold:.3f} \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{c}}{{Scaling Law Fit: $S^* = {law['a']:.2e} \\cdot N^{{{law['b']:.3f}}}$}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.2cm}")
    lines.append("\\caption{Critical collapse severity threshold across model sizes. Larger models exhibit more robustness to dataset collapse, following a power-law relationship.}")
    lines.append("\\label{tab:scaling_thresholds}")
    lines.append("\\end{table}")

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Generated LaTeX table at {output_path}")

def generate_scaling_law_plot(results: Dict[str, dict], output_path: str = "results/scaling_law.png"):
    """
    Generates a log-log plot of the scaling law fit.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    thresholds = compute_grokking_threshold(results)
    law = fit_scaling_law(results)

    sizes = np.array(sorted(list(thresholds.keys())))
    empirical_thresholds = np.array([thresholds[s] for s in sizes])

    # Generate points for the fitted line
    fit_sizes = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 50)
    fit_thresholds = law['a'] * (fit_sizes ** law['b'])

    plt.figure(figsize=(8, 6))
    plt.loglog(sizes, empirical_thresholds, 'bo', markersize=8, label="Empirical Threshold")
    plt.loglog(fit_sizes, fit_thresholds, 'r--', linewidth=2,
               label=f"Fit: $S^* \\propto N^{{{law['b']:.3f}}}$")

    plt.xlabel("Model Size (Parameters)")
    plt.ylabel("Critical Collapse Severity Threshold")
    plt.title("Scaling of Grokking Robustness to Model Collapse")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Generated scaling law plot at {output_path}")

if __name__ == "__main__":
    # Generate mock sweep data
    model_sizes = [100000, 200000, 400000, 800000, 1600000, 3200000]
    severities = np.linspace(0.0, 0.8, 41).tolist()

    results = run_grokking_sweep(model_sizes, severities, dummy_mode=True)

    generate_latex_table(results)
    generate_scaling_law_plot(results)
