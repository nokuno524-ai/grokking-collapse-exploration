"""
Analysis script to produce a boundary map (grokking success/onset vs (weight decay, learning rate) per optimizer).
Exports CSV/JSON and LaTeX tables.
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from opt_stats import load_sweep_results
except ImportError:
    from .opt_stats import load_sweep_results

def generate_boundary_map(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate a DataFrame with aggregated statistics per config."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Calculate success rate and mean grokking step per config
    agg_df = df.groupby(['optimizer', 'wd', 'lr', 'condition']).agg(
        n_seeds=('seed', 'count'),
        grokking_success_rate=('grokked', 'mean'),
        mean_grokking_step=('grokking_step', lambda x: np.mean([v for v in x if pd.notna(v) and v is not None]) if any(pd.notna(v) and v is not None for v in x) else np.nan),
        mean_final_acc=('final_test_acc', 'mean')
    ).reset_index()

    # Compute bootstrap CIs for grokking step
    def bootstrap_ci(group, n_boot=1000):
        steps = [r['grokking_step'] for _, r in group.iterrows() if r['grokked'] and r['grokking_step'] is not None]
        if len(steps) < 2:
            return pd.Series({'ci_low': np.nan, 'ci_high': np.nan})

        steps = np.array(steps)
        boot_means = [np.mean(np.random.choice(steps, size=len(steps), replace=True)) for _ in range(n_boot)]
        return pd.Series({
            'ci_low': np.percentile(boot_means, 2.5),
            'ci_high': np.percentile(boot_means, 97.5)
        })

    # Apply to each group
    cis = df.groupby(['optimizer', 'wd', 'lr', 'condition']).apply(bootstrap_ci).reset_index()

    # Merge CIs
    agg_df = pd.merge(agg_df, cis, on=['optimizer', 'wd', 'lr', 'condition'], how='left')

    return agg_df

def export_latex_table(df: pd.DataFrame, output_path: str):
    """Export boundary map to a LaTeX table."""
    if df.empty:
        return

    # Pivot to create a nice 2D table for each optimizer
    with open(output_path, 'w') as f:
        f.write("% Optimizer Boundary Map\n")

        for opt in df['optimizer'].unique():
            opt_df = df[df['optimizer'] == opt]

            f.write(f"\n% Optimizer: {opt}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{ll|ccc}\n")
            f.write("\\hline\n")
            f.write("WD & LR & Pure (Success \\%) & Collapsed (Success \\%) & $\\Delta$ Onset (Steps) \\\\\n")
            f.write("\\hline\n")

            for wd in sorted(opt_df['wd'].unique()):
                wd_df = opt_df[opt_df['wd'] == wd]

                for i, lr in enumerate(sorted(wd_df['lr'].unique())):
                    lr_df = wd_df[wd_df['lr'] == lr]

                    # Get pure stats
                    pure = lr_df[lr_df['condition'] == 'pure']
                    pure_succ = pure['grokking_success_rate'].values[0] * 100 if not pure.empty else 0
                    pure_step = pure['mean_grokking_step'].values[0] if not pure.empty else np.nan

                    # Get collapsed stats (take the first one if multiple, e.g. medium_collapse)
                    conds = [c for c in lr_df['condition'].unique() if c != 'pure']
                    if conds:
                        cond = lr_df[lr_df['condition'] == conds[0]]
                        cond_succ = cond['grokking_success_rate'].values[0] * 100 if not cond.empty else 0
                        cond_step = cond['mean_grokking_step'].values[0] if not cond.empty else np.nan
                    else:
                        cond_succ, cond_step = 0, np.nan

                    # Delta
                    if not np.isnan(pure_step) and not np.isnan(cond_step):
                        delta = f"{cond_step - pure_step:+.0f}"
                    else:
                        delta = "N/A"

                    wd_str = f"{wd}" if i == 0 else ""
                    f.write(f"{wd_str} & {lr} & {pure_succ:.0f}\\% & {cond_succ:.0f}\\% & {delta} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Grokking boundary map for {opt.upper()}}}\n")
            f.write(f"\\label{{tab:boundary_{opt}}}\n")
            f.write("\\end{table}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Generate boundary map")
    parser.add_argument("--results-dir", type=str, default="results/opt_sweep")
    parser.add_argument("--output-dir", type=str, default="analysis/optimizer")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_sweep_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")

        # Create empty DataFrame for testing
        df = pd.DataFrame(columns=['optimizer', 'wd', 'lr', 'condition', 'n_seeds',
                                  'grokking_success_rate', 'mean_grokking_step',
                                  'mean_final_acc', 'ci_low', 'ci_high'])

        df.to_csv(out_dir / "boundary_map.csv", index=False)
        with open(out_dir / "boundary_map.json", "w") as f:
            json.dump([], f)
        export_latex_table(df, str(out_dir / "boundary_map.tex"))
        return

    df = generate_boundary_map(results)

    # Export CSV
    df.to_csv(out_dir / "boundary_map.csv", index=False)

    # Export JSON
    with open(out_dir / "boundary_map.json", "w") as f:
        # Convert nan to null for JSON compatibility
        json_str = df.to_json(orient="records")
        json_data = json.loads(json_str)
        json.dump(json_data, f, indent=2)

    # Export LaTeX
    export_latex_table(df, str(out_dir / "boundary_map.tex"))

    print(f"Exported boundary map to {out_dir}")
    print(df.head())

if __name__ == "__main__":
    main()
