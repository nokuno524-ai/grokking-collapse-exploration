import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from src.analysis.utils import SEVERITY_ORDER
from src.analysis.cliff import extract_cliff_stats, permutation_test, cohen_d, compute_ci, trend_test

def analyze_seed(dir_path: Path) -> Dict[str, Any]:
    res_file = dir_path / "results.json"
    if not res_file.exists():
        return None
    try:
        data = json.load(open(res_file))
        if "history" not in data or not data["history"]:
            return None
        steps = np.array([row['step'] for row in data['history']])
        acc = np.array([row['test_acc'] for row in data['history']])
        return extract_cliff_stats(steps, acc)
    except Exception as e:
        print(f"Error reading {res_file}: {e}")
        return None

def generate_report(results_dir: Path, output_file: Path):
    seeds = [p.name for p in results_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not seeds:
        print("No seed directories found.")
        return

    stats = {sev: {'grokking_step': [], 'cliff_width': [], 'asymptotic_acc': [], 'r2': []} for sev in SEVERITY_ORDER}

    for seed in seeds:
        for sev in SEVERITY_ORDER:
            sev_dir = results_dir / seed / sev
            seed_stats = analyze_seed(sev_dir)
            if seed_stats:
                for k in stats[sev].keys():
                    stats[sev][k].append(seed_stats[k])

    lines = []
    lines.append("# Statistical Analysis of Grokking Cliffs\n")
    lines.append("This report analyzes the effect of model collapse severity on the grokking cliff.")
    lines.append("A logistic curve $y = \\text{bottom} + \\frac{\\text{top} - \\text{bottom}}{1 + e^{-k(x - x_0)}}$ is fit to the test accuracy vs. step curve for each run.\n")

    lines.append("## Cliff Statistics by Severity\n")
    lines.append("| Severity | N (Valid) | Grokking Step (95% CI) | Cliff Width (95% CI) | Asymptotic Acc (95% CI) | Mean R² |")
    lines.append("|---|---|---|---|---|---|")

    for sev in SEVERITY_ORDER:
        g_steps = np.array(stats[sev]['grokking_step'])
        g_steps = g_steps[~np.isnan(g_steps)]

        widths = np.array(stats[sev]['cliff_width'])
        widths = widths[~np.isnan(widths)]

        accs = np.array(stats[sev]['asymptotic_acc'])
        accs = accs[~np.isnan(accs)]

        r2s = np.array(stats[sev]['r2'])
        r2_mean = np.nanmean(r2s) if len(r2s) > 0 else np.nan

        n = len(g_steps)
        if n == 0:
            lines.append(f"| {sev} | 0 | - | - | - | {r2_mean:.2f} |")
            continue

        gm, gci = np.mean(g_steps), compute_ci(g_steps)
        wm, wci = np.mean(widths), compute_ci(widths)
        am, aci = np.mean(accs), compute_ci(accs)

        g_str = f"{gm:.0f} [{gci[0]:.0f}, {gci[1]:.0f}]" if not np.isnan(gci[0]) else f"{gm:.0f}"
        w_str = f"{wm:.0f} [{wci[0]:.0f}, {wci[1]:.0f}]" if not np.isnan(wci[0]) else f"{wm:.0f}"
        a_str = f"{am:.3f} [{aci[0]:.3f}, {aci[1]:.3f}]" if not np.isnan(aci[0]) else f"{am:.3f}"

        lines.append(f"| {sev} | {n} | {g_str} | {w_str} | {a_str} | {r2_mean:.2f} |")

    lines.append("\n## Hypothesis Tests\n")

    # Hypothesis: severity delays the cliff
    lines.append("### Effect of Severe Collapse on Grokking Step\n")
    pure_steps = np.array(stats['pure']['grokking_step'])
    severe_steps = np.array(stats['severe_collapse']['grokking_step'])

    if len(pure_steps[~np.isnan(pure_steps)]) > 0 and len(severe_steps[~np.isnan(severe_steps)]) > 0:
        p_val = permutation_test(pure_steps, severe_steps)
        d = cohen_d(pure_steps, severe_steps)
        lines.append(f"- **Comparison:** Pure vs Severe Collapse")
        lines.append(f"- **Permutation Test p-value:** {p_val:.4f}")
        lines.append(f"- **Cohen's d (effect size):** {d:.2f}")
    else:
        lines.append("- Not enough valid data points to compare Pure and Severe Collapse.")

    # Trend test
    lines.append("\n### Trend Analysis Across Severity Levels\n")
    trend_arrays = [np.array(stats[sev]['grokking_step']) for sev in SEVERITY_ORDER]
    p_trend = trend_test(trend_arrays)
    lines.append(f"- **Spearman trend test p-value (Grokking Step across severities):** {p_trend:.4f}")

    lines.append("\n## Caveats\n")
    lines.append("- Runs with no grokking (flat accuracy curves) are excluded from step and width calculations.")
    lines.append("- If N < 5, confidence intervals may be unreliable.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))

    print(f"Report generated at {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/multi_seed"))
    parser.add_argument("--output", type=Path, default=Path("analysis/statistical_report.md"))
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)
