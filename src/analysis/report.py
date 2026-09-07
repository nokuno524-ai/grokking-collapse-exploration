import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from .grok_detector.run_aggregator import load_runs_from_directories, align_runs_on_step
from .grok_detector.detectors import detect_cliffs, bootstrap_ci
from .grok_detector.stats import fit_severity_relationship, compare_endpoints

def generate_report(directories: List[str], output_dir: str, max_steps: int = 50000):
    """
    Generates the statistical report for grokking analysis.
    Outputs markdown, JSON, and PNG figures.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and group runs
    runs = load_runs_from_directories(directories)
    grouped = align_runs_on_step(runs)

    if not grouped:
        print("No valid runs found.")
        return

    # 2. Compute cliffs and statistics
    results = {}
    all_severities = []
    all_cliffs = []

    for severity in sorted(grouped.keys()):
        severity_runs = grouped[severity]

        cliffs = []
        censored_count = 0

        for run in severity_runs:
            cliff_info = detect_cliffs(run['history'], max_steps)
            if not cliff_info['is_censored']:
                cliffs.append(cliff_info['cliff_step'])
                all_severities.append(severity)
                all_cliffs.append(cliff_info['cliff_step'])
            else:
                censored_count += 1

        if cliffs:
            median, lower, upper = bootstrap_ci(np.array(cliffs))
        else:
            median, lower, upper = np.nan, np.nan, np.nan

        results[severity] = {
            'num_runs': len(severity_runs),
            'num_grokked': len(cliffs),
            'num_censored': censored_count,
            'cliffs': cliffs,
            'median_cliff': float(median) if not np.isnan(median) else None,
            'ci_lower': float(lower) if not np.isnan(lower) else None,
            'ci_upper': float(upper) if not np.isnan(upper) else None,
        }

    # 3. Fit severity vs cliff relationship (ignoring censored for the fit)
    fit_results = fit_severity_relationship(np.array(all_severities), np.array(all_cliffs))

    # 4. Compare endpoints
    baseline = min(grouped.keys()) if grouped else 0.0
    endpoint_comparisons = compare_endpoints(grouped, baseline_severity=baseline)

    # 5. Generate JSON
    report_data = {
        'severity_stats': results,
        'fits': fit_results,
        'endpoint_comparisons': endpoint_comparisons
    }
    with open(out_path / 'grokking_stats.json', 'w') as f:
        json.dump(report_data, f, indent=2)

    # 6. Generate Figures
    _plot_violins(results, out_path / 'cliff_violins.png')
    _plot_accuracy_curves(grouped, results, max_steps, out_path / 'accuracy_curves.png')

    # 7. Generate Markdown Report
    _write_markdown(report_data, out_path / 'grokking_report.md')

def _plot_violins(results: Dict[float, Any], filepath: Path):
    severities = sorted(results.keys())
    data = []
    labels = []

    for sev in severities:
        if results[sev]['cliffs']:
            data.append(results[sev]['cliffs'])
            labels.append(f"{sev}\n(n={results[sev]['num_grokked']})")

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Severity (Noise Fraction / Collapse)')
    ax.set_ylabel('Grokking Cliff Step')
    ax.set_title('Distribution of Grokking Cliff Steps by Severity')
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)

def _plot_accuracy_curves(grouped: Dict[float, List[Dict[str, Any]]], results: Dict[float, Any], max_steps: int, filepath: Path):
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))

    for (severity, runs), color in zip(sorted(grouped.items()), colors):
        if not runs: continue

        # We need a common grid to interpolate if steps don't align perfectly
        # But usually in this repo, steps align exactly.
        all_steps = sorted(list(set(step for run in runs for step in [r['step'] for r in run['history']])))
        if not all_steps: continue

        acc_matrix = []
        for run in runs:
            # map step -> acc
            step_to_acc = {r['step']: r['test_acc'] for r in run['history']}
            acc_arr = [step_to_acc.get(s, np.nan) for s in all_steps]
            acc_matrix.append(acc_arr)

        acc_matrix = np.array(acc_matrix)

        # Calculate mean and percentiles
        with np.errstate(invalid='ignore'):
            mean_acc = np.nanmean(acc_matrix, axis=0)
            p25 = np.nanpercentile(acc_matrix, 25, axis=0)
            p75 = np.nanpercentile(acc_matrix, 75, axis=0)

        ax.plot(all_steps, mean_acc, label=f'Severity {severity}', color=color)
        ax.fill_between(all_steps, p25, p75, color=color, alpha=0.2)

        # Annotate median cliff if it exists
        med_cliff = results[severity]['median_cliff']
        if med_cliff is not None:
            ax.axvline(med_cliff, color=color, linestyle='--', alpha=0.5)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Aggregated Test Accuracy Trajectories with Median Grokking Cliffs')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)

def _write_markdown(data: Dict[str, Any], filepath: Path):
    lines = [
        "# Grokking Statistical Analysis Report",
        "",
        "## Grokking Cliff Detection by Severity",
        "",
        "| Severity | Runs | Grokked | Censored | Median Cliff | 95% CI Lower | 95% CI Upper |",
        "|----------|------|---------|----------|--------------|--------------|--------------|"
    ]

    for sev, stats in data['severity_stats'].items():
        med = f"{stats['median_cliff']:.1f}" if stats['median_cliff'] is not None else "N/A"
        low = f"{stats['ci_lower']:.1f}" if stats['ci_lower'] is not None else "N/A"
        up = f"{stats['ci_upper']:.1f}" if stats['ci_upper'] is not None else "N/A"

        lines.append(f"| {sev} | {stats['num_runs']} | {stats['num_grokked']} | {stats['num_censored']} | {med} | {low} | {up} |")

    lines.extend([
        "",
        "## Curve Fits (Severity vs. Cliff Step)",
        ""
    ])

    if 'linear' in data['fits']:
        lin = data['fits']['linear']
        lines.append(f"- **Linear Fit:** $R^2 = {lin['r2']:.3f}$, $p = {lin['p_value']:.2e}$")

    if 'logistic' in data['fits']:
        log = data['fits']['logistic']
        lines.append(f"- **Logistic Fit:** $R^2 = {log['r2']:.3f}$")

    lines.extend([
        "",
        "## Endpoint Accuracy Comparison",
        f"**Baseline Severity:** {data['endpoint_comparisons'].get('baseline_severity', 'N/A')}",
        "",
        "| Severity | Mean Final Acc | Median Final Acc | Holm-Adj p-value |",
        "|----------|----------------|------------------|------------------|"
    ])

    if 'comparisons' in data['endpoint_comparisons']:
        for sev, comp in data['endpoint_comparisons']['comparisons'].items():
            lines.append(f"| {sev} | {comp['mean_acc']:.3f} | {comp['median_acc']:.3f} | {comp['adj_p_value']:.2e} |")

    lines.extend([
        "",
        "## Visualizations",
        "![Cliff Violins](cliff_violins.png)",
        "",
        "![Accuracy Curves](accuracy_curves.png)"
    ])

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True, help='Directories containing run logs')
    parser.add_argument('--out', required=True, help='Output directory for report')
    parser.add_argument('--max-steps', type=int, default=50000, help='Max training steps')
    args = parser.parse_args()

    generate_report(args.dirs, args.out, args.max_steps)
