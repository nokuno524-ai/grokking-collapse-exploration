import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import scipy.stats as stats

def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Recursively loads all results.json files from the results directory.

    Args:
        results_dir: Root directory to search for results.

    Returns:
        DataFrame containing summary statistics for all found experiments.
    """
    all_data = []

    for root, _, files in os.walk(results_dir):
        if "results.json" in files:
            file_path = os.path.join(root, "results.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Flatten the data structure
                config = data.get('config', {})

                # Default values for missing keys
                condition = config.get('condition_name', os.path.basename(root))

                # Get weight norm change from history
                history = data.get('history', [])
                wn_change = 0.0
                if history:
                    first_wn = history[0].get('weight_norm', 0.0)
                    last_wn = history[-1].get('weight_norm', 0.0)
                    wn_change = last_wn - first_wn

                row = {
                    'path': file_path,
                    'condition': condition,
                    'collapse_level': config.get('collapse_level', 0.0),
                    'noise_fraction': config.get('noise_fraction', 0.0),
                    'weight_decay': config.get('weight_decay', 1.0),
                    'seed': config.get('seed', 42),
                    'grokked': data.get('grokked', False),
                    'grok_step': data.get('grokking_step', -1),
                    'final_train_acc': data.get('final_train_acc', 0.0),
                    'final_test_acc': data.get('final_test_acc', 0.0),
                    'final_weight_norm': data.get('final_weight_norm', 0.0),
                    'weight_norm_change': wn_change
                }
                all_data.append(row)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Standardize condition names for aggregation if not already descriptive
    if 'pure' in df['path'].values[0]: # Crude heuristic, improve based on actual data
        df['standard_condition'] = df.apply(
            lambda x: f"noise_{x['noise_fraction']}_wd_{x['weight_decay']}" if 'exp_c_grid' in x['path'] else
                      ('pure' if x['collapse_level'] == 0.0 else f"collapse_{x['collapse_level']}"),
            axis=1
        )
    else:
        # Fallback
        df['standard_condition'] = df['condition']

    return df

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary statistics grouped by condition.
    """
    if df.empty:
        return pd.DataFrame()

    # Group by standard condition
    summary = df.groupby('standard_condition').agg({
        'grokked': ['mean', 'count'],
        'grok_step': ['mean', 'std', lambda x: x[x != -1].mean()], # mean only for those that grokked
        'final_test_acc': ['mean', 'std'],
        'weight_norm_change': ['mean', 'std']
    }).reset_index()

    # Rename columns to be flat
    summary.columns = ['Condition', 'Grok_Rate', 'N_Runs', 'Mean_Grok_Step_All', 'Std_Grok_Step',
                       'Mean_Grok_Step_Success', 'Mean_Final_Acc', 'Std_Final_Acc',
                       'Mean_WN_Change', 'Std_WN_Change']

    # Clean up Mean_Grok_Step_Success NaN values
    summary['Mean_Grok_Step_Success'] = summary['Mean_Grok_Step_Success'].fillna(-1)

    return summary

if __name__ == "__main__":
    pass

def test_statistical_significance(df: pd.DataFrame, pure_cond: str, collapse_cond: str, metric: str = 'final_test_acc') -> Dict[str, Any]:
    """
    Performs Mann-Whitney U test between two conditions.
    """
    group_pure = df[df['standard_condition'] == pure_cond][metric].values
    group_collapse = df[df['standard_condition'] == collapse_cond][metric].values

    if len(group_pure) == 0 or len(group_collapse) == 0:
        return {'error': 'One or both groups empty'}

    try:
        stat, p_value = stats.mannwhitneyu(group_pure, group_collapse, alternative='two-sided')
        return {
            'metric': metric,
            'pure_cond': pure_cond,
            'collapse_cond': collapse_cond,
            'u_statistic': float(stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'pure_mean': float(np.mean(group_pure)),
            'collapse_mean': float(np.mean(group_collapse))
        }
    except Exception as e:
        return {'error': str(e)}

def plot_comparison(df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: str):
    """
    Generates a box plot comparing a metric across conditions.
    """
    if df.empty:
        return

    plt.figure(figsize=(10, 6))

    # Sort conditions for consistent plotting
    conditions = sorted(df['standard_condition'].unique())
    data_to_plot = [df[df['standard_condition'] == cond][metric].dropna().values for cond in conditions]

    plt.boxplot(data_to_plot, tick_labels=conditions)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_learning_curves(df: pd.DataFrame, results_dir: str, output_path: str):
    """
    Plots average training/test curves across conditions.
    This requires re-reading the full history from files.
    """
    if df.empty:
        return

    plt.figure(figsize=(12, 5))

    # Plot test accuracy
    plt.subplot(1, 2, 1)

    conditions = sorted(df['standard_condition'].unique())

    for cond in conditions:
        cond_df = df[df['standard_condition'] == cond]
        if len(cond_df) > 0:
            # Just take the first run for visualization simplicity
            # In a full robust implementation, we would average across runs
            first_run_path = cond_df.iloc[0]['path']
            try:
                with open(first_run_path, 'r') as f:
                    data = json.load(f)
                history = data.get('history', [])
                if history:
                    steps = [h['step'] for h in history]
                    test_acc = [h.get('test_acc', 0) for h in history]
                    plt.plot(steps, test_acc, label=cond)
            except Exception:
                pass

    plt.xlabel('Steps')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Trajectories (Sample Runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot weight norm
    plt.subplot(1, 2, 2)
    for cond in conditions:
        cond_df = df[df['standard_condition'] == cond]
        if len(cond_df) > 0:
            first_run_path = cond_df.iloc[0]['path']
            try:
                with open(first_run_path, 'r') as f:
                    data = json.load(f)
                history = data.get('history', [])
                if history:
                    steps = [h['step'] for h in history]
                    wn = [h.get('weight_norm', 0) for h in history]
                    plt.plot(steps, wn, label=cond)
            except Exception:
                pass

    plt.xlabel('Steps')
    plt.ylabel('Weight Norm')
    plt.title('Weight Norm Trajectories (Sample Runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_markdown_report(summary_df: pd.DataFrame, sig_results: List[Dict], report_path: str):
    """
    Generates a Markdown report summarizing the findings.
    """
    with open(report_path, 'w') as f:
        f.write("# Grokking & Model Collapse: Analysis Results\n\n")

        f.write("## Summary Statistics\n\n")

        if not summary_df.empty:
            # Format numbers for Markdown table
            display_df = summary_df.copy()
            for col in display_df.select_dtypes(include=[float]).columns:
                display_df[col] = display_df[col].map('{:.4f}'.format)

            f.write(display_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Statistical Significance\n\n")
        if sig_results:
            f.write("| Metric | Pure Condition | Collapse Condition | P-Value | Significant |\n")
            f.write("|---|---|---|---|---|\n")
            for res in sig_results:
                if 'error' not in res:
                    sig_str = "Yes" if res['significant'] else "No"
                    f.write(f"| {res['metric']} | {res['pure_cond']} | {res['collapse_cond']} | {res['p_value']:.4e} | {sig_str} |\n")
            f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("### Accuracy Distribution\n")
        f.write("![Final Test Accuracy](accuracy_comparison.png)\n\n")

        f.write("### Learning Trajectories\n")
        f.write("![Learning Curves](learning_curves.png)\n\n")

def run_analysis(results_dir: str = "results", output_dir: str = "results"):
    """
    Main entry point for running the complete analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    df = load_all_results(results_dir)

    if df.empty:
        print("No results found. Analysis aborted.")
        return

    print("Generating summary statistics...")
    summary_df = generate_summary_statistics(df)

    print("Plotting comparisons...")
    plot_comparison(df, 'final_test_acc', 'Final Test Accuracy', 'Final Test Accuracy by Condition',
                    os.path.join(output_dir, 'accuracy_comparison.png'))

    print("Plotting trajectories...")
    plot_learning_curves(df, results_dir, os.path.join(output_dir, 'learning_curves.png'))

    print("Testing statistical significance...")
    sig_results = []
    # Identify pure and a collapse condition for test if possible
    conditions = df['standard_condition'].unique()

    pure_c = None
    collapse_c = None

    # Try to find default pure and noise>0 cases
    for c in conditions:
        if 'pure' in c or 'noise_0.0_' in c:
            pure_c = c
        elif 'collapse_' in c or 'noise_' in c and '0.0' not in c:
            collapse_c = c

    if pure_c and collapse_c:
        sig_results.append(test_statistical_significance(df, pure_c, collapse_c, 'final_test_acc'))
        sig_results.append(test_statistical_significance(df, pure_c, collapse_c, 'grok_step'))

    print("Generating markdown report...")
    generate_markdown_report(summary_df, sig_results, os.path.join(output_dir, 'ANALYSIS.md'))
    print(f"Analysis complete. Report saved to {os.path.join(output_dir, 'ANALYSIS.md')}")

if __name__ == "__main__":
    run_analysis()
