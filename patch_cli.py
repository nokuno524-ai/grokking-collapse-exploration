import re

with open("src/cli_data_degradation.py", "r") as f:
    content = f.read()

# Add missing imports for correlation
import_str = "import scipy.stats\nimport glob\nfrom src.analysis.data_degradation import compute_data_degradation_metrics"
content = content.replace("from src.analysis.data_degradation import compute_data_degradation_metrics", import_str)

# Add correlation analysis logic
correlation_func = """
def load_grokking_outcomes(seed: int) -> Dict[str, Any]:
    \"\"\"Loads grokking step/failure from results directories for the given seed.\"\"\"
    outcomes = {}
    base_dir = f"results/multi_seed/{seed}"
    if not os.path.exists(base_dir):
        return outcomes

    for condition in os.listdir(base_dir):
        results_path = os.path.join(base_dir, condition, "results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                outcomes[condition] = {
                    'grok_step': data.get('grok_step'),
                    'grok_failure': data.get('grok_step') is None
                }
    return outcomes

def compute_correlations(df: pd.DataFrame, outcomes: Dict[str, Any]) -> pd.DataFrame:
    \"\"\"Computes correlations between data degradation metrics and grokking failure.\"\"\"
    if not outcomes:
        print("No grokking outcomes found. Skipping correlation analysis.")
        return pd.DataFrame()

    # We correlate metrics at max depth for each condition with the outcome
    max_depth_df = df.loc[df.groupby('condition')['generation'].idxmax()].copy()

    # Merge outcomes
    max_depth_df['grok_failure'] = max_depth_df['condition'].map(lambda c: outcomes.get(c, {}).get('grok_failure', False)).astype(int)
    max_depth_df['grok_step'] = max_depth_df['condition'].map(lambda c: outcomes.get(c, {}).get('grok_step', float('nan')))

    metrics = ['entropy', 'distinct_1', 'distinct_2', 'compression_ratio',
               'duplicate_fraction', 'js_divergence_from_base']

    correlations = []
    for metric in metrics:
        # Spearman correlation with failure
        failure_corr, failure_pval = scipy.stats.spearmanr(max_depth_df[metric], max_depth_df['grok_failure'])

        # Spearman correlation with grok step (for non-failures)
        valid_steps = max_depth_df.dropna(subset=['grok_step'])
        if len(valid_steps) > 1:
            step_corr, step_pval = scipy.stats.spearmanr(valid_steps[metric], valid_steps['grok_step'])
        else:
            step_corr, step_pval = float('nan'), float('nan')

        correlations.append({
            'Metric': metric,
            'Corr w/ Failure': failure_corr,
            'Failure p-value': failure_pval,
            'Corr w/ Grok Step': step_corr,
            'Grok Step p-value': step_pval
        })

    corr_df = pd.DataFrame(correlations)
    return corr_df

"""
content = content.replace("def run_sweep_and_compute_metrics", correlation_func + "\ndef run_sweep_and_compute_metrics")

# Call correlation analysis in the main run function
main_changes = """
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'data_degradation_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    # Load outcomes and compute correlations
    outcomes = load_grokking_outcomes(seed)
    corr_df = compute_correlations(df, outcomes)

    if not corr_df.empty:
        corr_csv_path = os.path.join(output_dir, 'metric_grokking_correlations.csv')
        corr_df.to_csv(corr_csv_path, index=False)
        print(f"Saved correlations to {corr_csv_path}")
        print("\\nCorrelation Table:")
        print(corr_df.to_markdown(index=False))
"""
content = re.sub(r"# Save to CSV.*?(?=# Generate plots)", main_changes, content, flags=re.DOTALL)

with open("src/cli_data_degradation.py", "w") as f:
    f.write(content)
