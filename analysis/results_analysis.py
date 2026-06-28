import json
import glob
import os
import pandas as pd
import numpy as np
from scipy import stats

def load_results(base_dir="results"):
    files = glob.glob(os.path.join(base_dir, "**", "results.json"), recursive=True)
    data = []
    for f in files:
        try:
            with open(f, 'r') as file:
                d = json.load(file)
                config = d.get('config', {})
                # Some conditions use noise_fraction, some use collapse_severity/collapse_level
                collapse_level = config.get('collapse_level', 0.0)
                noise_fraction = config.get('noise_fraction', 0.0)

                # Determine collapse condition for ANOVA
                if 'grid' in f or 'multi_seed' in f:
                    condition = "unknown"
                    if "pure" in f or "noise0/" in f or "noise0.0/" in f or "level0_sev" in f:
                        condition = "pure"
                    elif "low_collapse" in f or "level0.05" in f:
                        condition = "low_collapse"
                    elif "medium_collapse" in f or "level0.15" in f or "noise0.15" in f:
                        condition = "medium_collapse"
                    elif "high_collapse" in f or "level0.3" in f:
                        condition = "high_collapse"
                    elif "severe_collapse" in f or "level0.5" in f:
                        condition = "severe_collapse"
                else:
                    if "pure" in f: condition = "pure"
                    elif "low_collapse" in f: condition = "low_collapse"
                    elif "medium_collapse" in f: condition = "medium_collapse"
                    elif "high_collapse" in f: condition = "high_collapse"
                    elif "severe_collapse" in f: condition = "severe_collapse"
                    elif "noise_baseline" in f: condition = "noise_baseline"
                    elif "scarcity_baseline" in f: condition = "scarcity_baseline"
                    else: condition = "other"

                if "noise0.15" in f and "wd0.3" in f:
                    condition = "medium_collapse"
                if "noise0" in f and "wd0.3" in f:
                    condition = "pure"

                # Calculate weight norm change if history exists
                history = d.get('history', [])
                weight_norm_change = np.nan
                if history and len(history) > 0:
                    first_norm = history[0].get('weight_norm', np.nan)
                    last_norm = history[-1].get('weight_norm', np.nan)
                    if pd.notna(first_norm) and pd.notna(last_norm):
                        weight_norm_change = last_norm - first_norm

                row = {
                    'file': f,
                    'condition': condition,
                    'collapse_level': collapse_level,
                    'noise_fraction': noise_fraction,
                    'seed': config.get('seed', -1),
                    'grokked': float(d.get('grokked', False)),
                    'grokking_step': d.get('grokking_step', np.nan),
                    'final_test_acc': d.get('final_test_acc', np.nan),
                    'final_train_acc': d.get('final_train_acc', np.nan),
                    'final_weight_norm': d.get('final_weight_norm', np.nan),
                    'weight_norm_change': weight_norm_change,
                    'final_fourier_concentration': d.get('final_fourier_concentration', np.nan),
                    'final_embedding_rank': d.get('final_embedding_rank', np.nan),
                    'severity_metric': max(collapse_level, noise_fraction)
                }
                data.append(row)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.DataFrame(data)

def run_analysis(df):
    results = {}
    print("--- RESULTS ANALYSIS ---")

    # 1. Compare grokking rates: Pure vs Collapsed
    pure_df = df[df['severity_metric'] == 0]
    collapsed_df = df[df['severity_metric'] > 0]

    pure_grok_rate = pure_df['grokked'].mean()
    coll_grok_rate = collapsed_df['grokked'].mean()
    print(f"Pure Grokking Rate: {pure_grok_rate:.2f} (n={len(pure_df)})")
    print(f"Collapsed Grokking Rate: {coll_grok_rate:.2f} (n={len(collapsed_df)})")

    if len(pure_df) > 0 and len(collapsed_df) > 0:
        # T-test for grokking rate
        t_stat, p_val_t = stats.ttest_ind(pure_df['grokked'], collapsed_df['grokked'], equal_var=False)
        print(f"T-test (Pure vs Collapsed Grokking): t={t_stat:.4f}, p={p_val_t:.4e}")

        # Mann-Whitney U test
        u_stat, p_val_u = stats.mannwhitneyu(pure_df['grokked'], collapsed_df['grokked'], alternative='two-sided')
        print(f"Mann-Whitney U (Pure vs Collapsed Grokking): U={u_stat:.4f}, p={p_val_u:.4e}")

    # 2. Correlation between collapse severity and metrics
    print("\n--- CORRELATIONS WITH SEVERITY ---")
    metrics = ['grokking_step', 'final_test_acc', 'weight_norm_change', 'final_fourier_concentration']
    for m in metrics:
        valid_df = df.dropna(subset=['severity_metric', m])
        if len(valid_df) > 2:
            r, p = stats.pearsonr(valid_df['severity_metric'], valid_df[m])
            print(f"Severity vs {m}: r={r:.4f}, p={p:.4e} (n={len(valid_df)})")
        else:
            print(f"Severity vs {m}: Not enough data")

    # 3. ANOVA across collapse conditions
    print("\n--- ANOVA ACROSS CONDITIONS ---")
    conditions = ['pure', 'low_collapse', 'medium_collapse', 'high_collapse', 'severe_collapse']

    # Check if we have enough data for ANOVA
    acc_groups = [df[df['condition'] == c]['final_test_acc'].dropna() for c in conditions]
    acc_groups = [g for g in acc_groups if len(g) > 0]

    if len(acc_groups) >= 2:
        f_stat, p_val_f = stats.f_oneway(*acc_groups)
        print(f"ANOVA (Final Test Acc across {len(acc_groups)} conditions): F={f_stat:.4f}, p={p_val_f:.4e}")

    fourier_groups = [df[df['condition'] == c]['final_fourier_concentration'].dropna() for c in conditions]
    fourier_groups = [g for g in fourier_groups if len(g) > 0]

    if len(fourier_groups) >= 2:
        f_stat, p_val_f = stats.f_oneway(*fourier_groups)
        print(f"ANOVA (Fourier Concentration across {len(fourier_groups)} conditions): F={f_stat:.4f}, p={p_val_f:.4e}")

    return df

def generate_latex_table(df, output_path="analysis/results_table.tex"):
    print(f"\nGenerating LaTeX table to {output_path}...")

    # Group by severity/condition
    summary = df.groupby('condition').agg({
        'grokked': ['mean', 'count'],
        'grokking_step': 'mean',
        'final_test_acc': ['mean', 'std'],
        'final_fourier_concentration': ['mean', 'std'],
        'weight_norm_change': ['mean', 'std']
    }).reset_index()

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Summary of Grokking Metrics Across Collapse Conditions}\n"
    latex += "\\label{tab:results_summary}\n"
    latex += "\\begin{tabular}{lcccccc}\n"
    latex += "\\hline\n"
    latex += "Condition & Grok Rate & N & Delay & Test Acc & Fourier Conc. & $\\Delta$ Weight Norm \\\\\n"
    latex += "\\hline\n"

    conditions = ['pure', 'low_collapse', 'medium_collapse', 'high_collapse', 'severe_collapse', 'noise_baseline', 'scarcity_baseline']
    for cond in conditions:
        row = summary[summary['condition'] == cond]
        if len(row) > 0:
            row = row.iloc[0]
            grok_rate = f"{row[('grokked', 'mean')]:.2f}"
            n = int(row[('grokked', 'count')])

            delay = row[('grokking_step', 'mean')]
            delay_str = f"{delay:.0f}" if pd.notna(delay) else "-"

            acc = row[('final_test_acc', 'mean')]
            acc_std = row[('final_test_acc', 'std')]
            acc_str = f"{acc:.2f} $\\pm$ {acc_std:.2f}" if pd.notna(acc_std) else f"{acc:.2f}"

            fourier = row[('final_fourier_concentration', 'mean')]
            fourier_std = row[('final_fourier_concentration', 'std')]
            fourier_str = f"{fourier:.3f} $\\pm$ {fourier_std:.3f}" if pd.notna(fourier_std) else f"{fourier:.3f}"

            wn = row[('weight_norm_change', 'mean')]
            wn_std = row[('weight_norm_change', 'std')]
            wn_str = f"{wn:.1f} $\\pm$ {wn_std:.1f}" if pd.notna(wn_std) else f"{wn:.1f}"

            latex += f"{cond.replace('_', ' ').title()} & {grok_rate} & {n} & {delay_str} & {acc_str} & {fourier_str} & {wn_str} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    with open(output_path, 'w') as f:
        f.write(latex)
    print("Table generation complete.")

if __name__ == "__main__":
    df = load_results()
    run_analysis(df)
    generate_latex_table(df)
