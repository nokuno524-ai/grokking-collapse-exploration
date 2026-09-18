import json
from pathlib import Path
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import pandas as pd

def compute_ci(data, confidence=0.95):
    """Compute mean and confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    if n < 2:
        return m, 0.0
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def load_results(base_dir, conditions):
    """Load results for multiple seeds and conditions."""
    results = {cond: [] for cond in conditions}
    base_path = Path(base_dir)

    if not base_path.exists():
        return results

    for seed_dir in base_path.iterdir():
        if not seed_dir.is_dir():
            continue

        for cond in conditions:
            res_file = seed_dir / cond / "results.json"
            if res_file.exists():
                with open(res_file) as f:
                    results[cond].append(json.load(f))

    return results

def aggregate_and_report(base_dir, conditions, metric="final_test_acc", baseline_cond=None):
    """Aggregate across seeds, compute stats and multiple comparison correction."""
    results = load_results(base_dir, conditions)

    summary = []
    baseline_data = None
    if baseline_cond and baseline_cond in results:
        baseline_data = [r[metric] for r in results[baseline_cond] if r[metric] is not None]

    p_values = []
    cond_names = []

    for cond in conditions:
        data = [r[metric] for r in results[cond] if metric in r and r[metric] is not None]

        if not data:
            summary.append({"condition": cond, "mean": float("nan"), "ci": float("nan"), "n": 0})
            continue

        m, ci = compute_ci(data)

        # Test against baseline if provided
        p_val = float("nan")
        if baseline_data and cond != baseline_cond and len(data) > 1 and len(baseline_data) > 1:
            # Add small noise to avoid identical data issues with t-test
            jittered_data = np.array(data) + np.random.normal(0, 1e-10, len(data))
            jittered_base = np.array(baseline_data) + np.random.normal(0, 1e-10, len(baseline_data))
            _, p_val = stats.ttest_ind(jittered_data, jittered_base, equal_var=False)
            if not np.isnan(p_val):
                p_values.append(p_val)
                cond_names.append(cond)

        summary.append({
            "condition": cond,
            "mean": m,
            "ci": ci,
            "n": len(data),
            "p_val": p_val
        })

    # Apply Holm correction
    if p_values:
        reject, pvals_corrected, _, _ = multipletests(p_values, method='holm')

        for i, cond in enumerate(cond_names):
            for row in summary:
                if row["condition"] == cond:
                    row["p_val_adj"] = pvals_corrected[i]
                    row["significant"] = reject[i]

    return pd.DataFrame(summary)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Base directory for results")
    parser.add_argument("--conditions", type=str, required=True, help="Comma separated conditions")
    parser.add_argument("--metric", type=str, default="final_test_acc")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--out", type=str, default="report.md")

    args = parser.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",")]

    df = aggregate_and_report(args.dir, conditions, args.metric, args.baseline)
    print(df.to_markdown(index=False))

    with open(args.out, "w") as f:
        f.write(f"# Results for {args.dir} ({args.metric})\n\n")
        f.write(df.to_markdown(index=False))
