import json
import os
import pandas as pd
import numpy as np
from glob import glob

def aggregate_results(results_base_dir):
    """
    Load all experiment results, compute summary statistics, and output a LaTeX table.
    """
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    data = []

    for cond in conditions:
        cond_dir = os.path.join(results_base_dir, cond)
        if not os.path.exists(cond_dir):
            continue

        seed_dirs = glob(os.path.join(cond_dir, "seed_*"))

        # Override pure title
        cond_label = "Pure Data" if cond == "pure" else cond.replace('_', ' ').title()

        if not seed_dirs:
            # Handle case where results are directly in cond_dir (no seed subdirectories)
            json_path = os.path.join(cond_dir, "results.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    res = json.load(f)

                data.append({
                    'Condition': cond_label,
                    'Seed': 'seed_0',
                    'Grokked': res.get('grokked', False),
                    'Grokking Step': res.get('grokking_step', 50000),
                    'Final Test Acc': res.get('final_test_acc', 0.0),
                    'Final Train Acc': res.get('final_train_acc', 0.0),
                    'Final Weight Norm': res.get('final_weight_norm', np.nan)
                })
        else:
            for seed_dir in seed_dirs:
                json_path = os.path.join(seed_dir, "results.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        res = json.load(f)

                    data.append({
                    'Condition': cond_label,
                        'Seed': os.path.basename(seed_dir),
                        'Grokked': res.get('grokked', False),
                        'Grokking Step': res.get('grokking_step', 50000),
                        'Final Test Acc': res.get('final_test_acc', 0.0),
                        'Final Train Acc': res.get('final_train_acc', 0.0),
                        'Final Weight Norm': res.get('final_weight_norm', np.nan)
                    })

    if not data:
        return "No data found."

    df = pd.DataFrame(data)

    # Calculate summary statistics
    summary = df.groupby('Condition').agg(
        Grok_Rate=('Grokked', 'mean'),
        Mean_Grok_Step=('Grokking Step', 'mean'),
        Std_Grok_Step=('Grokking Step', 'std'),
        Mean_Test_Acc=('Final Test Acc', 'mean'),
        Std_Test_Acc=('Final Test Acc', 'std'),
        Mean_Weight_Norm=('Final Weight Norm', 'mean')
    ).reset_index()

    # Format strings
    summary['Grok_Rate'] = (summary['Grok_Rate'] * 100).astype(int).astype(str) + '%'
    summary['Grok_Step'] = summary.apply(lambda row: f"{row['Mean_Grok_Step']:.0f} ± {row['Std_Grok_Step']:.0f}" if pd.notna(row['Std_Grok_Step']) else f"{row['Mean_Grok_Step']:.0f}", axis=1)
    summary['Test_Acc'] = summary.apply(lambda row: f"{row['Mean_Test_Acc']*100:.1f} ± {row['Std_Test_Acc']*100:.1f}%" if pd.notna(row['Std_Test_Acc']) else f"{row['Mean_Test_Acc']*100:.1f}%", axis=1)
    summary['Weight_Norm'] = summary.apply(lambda row: f"{row['Mean_Weight_Norm']:.2f}", axis=1)

    # Select columns for table
    table_df = summary[['Condition', 'Grok_Rate', 'Grok_Step', 'Test_Acc', 'Weight_Norm']]
    table_df.columns = ['Condition', 'Grokking Rate', 'Grokking Step', 'Final Test Accuracy', 'Final Weight Norm']

    # Sort to ensure logical order
    cond_order = {'Pure Data': 0, 'Low Collapse': 1, 'Medium Collapse': 2, 'High Collapse': 3, 'Severe Collapse': 4}
    table_df['order'] = table_df['Condition'].map(cond_order)
    table_df = table_df.sort_values('order').drop('order', axis=1)

    # Convert to LaTeX
    latex_table = table_df.to_latex(index=False, escape=False)

    # Add caption and label
    latex_out = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"{latex_table}"
        "\\caption{Summary of grokking performance across collapse conditions. "
        "Grokking rate is the percentage of seeds that reached $>90\\%$ test accuracy.}\n"
        "\\label{tab:main_results}\n"
        "\\end{table}\n"
    )

    return latex_out

if __name__ == "__main__":
    latex_code = aggregate_results("results")
    print(latex_code)

    # Save to file
    with open("analysis/results_table.tex", "w") as f:
        f.write(latex_code)
    print("Saved to analysis/results_table.tex")
