import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

import scipy.stats
import glob
from src.analysis.data_degradation import compute_data_degradation_metrics
from src.data import DatasetConfig, generate_modular_arithmetic, get_all_conditions


def load_grokking_outcomes(seed: int) -> Dict[str, Any]:
    """Loads grokking step/failure from results directories for the given seed."""
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
    """Computes correlations between data degradation metrics and grokking failure."""
    if not outcomes:
        print("\nNo grokking outcomes found. Skipping correlation analysis.")
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


def run_sweep_and_compute_metrics(output_dir: str, max_depth: int, seed: int = 42):
    """Generates recursive datasets and computes degradation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = get_all_conditions(seed=seed)
    vocab_size = conditions['pure'].prime

    results = []

    for condition_name, config in conditions.items():
        print(f"Processing condition: {condition_name}")

        # Get base (depth=0) corpus for this condition (pure)
        base_config = DatasetConfig(
            prime=config.prime, train_fraction=config.train_fraction,
            collapse_level=0.0, collapse_severity=0.0, seed=seed
        )
        _, base_targets, _, _ = generate_modular_arithmetic(base_config)
        base_corpus = base_targets.tolist()

        for depth in range(max_depth + 1):
            # Generate dataset for this depth
            depth_config = DatasetConfig(
                prime=config.prime,
                train_fraction=config.train_fraction,
                collapse_level=config.collapse_level,
                collapse_severity=config.collapse_severity,
                seed=seed,
                # Use tree_depth=depth for our modified data.py
            )
            # Monkey-patch tree_depth
            depth_config.tree_depth = depth

            _, targets, _, _ = generate_modular_arithmetic(depth_config)
            corpus = targets.tolist()

            # Compute metrics
            metrics = compute_data_degradation_metrics(corpus, base_corpus, vocab_size)
            metrics['condition'] = condition_name
            metrics['generation'] = depth
            metrics['seed'] = seed

            results.append(metrics)


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
        print("\nCorrelation Table:")
        print(corr_df.to_markdown(index=False))
# Generate plots
    plot_metrics(df, output_dir)

    return df

def plot_metrics(df: pd.DataFrame, output_dir: str):
    """Generates plots for degradation metrics vs generation."""
    metrics_to_plot = ['entropy', 'distinct_1', 'distinct_2', 'compression_ratio',
                       'duplicate_fraction', 'js_divergence_from_base']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        for condition in df['condition'].unique():
            condition_df = df[df['condition'] == condition]
            plt.plot(condition_df['generation'], condition_df[metric], marker='o', label=condition)

        plt.title(f'{metric.replace("_", " ").title()} vs Generation')
        plt.xlabel('Generation (Depth)')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'{metric}_vs_generation.png')
        plt.savefig(plot_path)
        plt.close()

    print(f"Saved plots to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compute data degradation metrics across recursive generations.")
    parser.add_argument("--output-dir", type=str, default="analysis/data_degradation", help="Directory to save outputs")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum recursion depth to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_sweep_and_compute_metrics(args.output_dir, args.max_depth, args.seed)

if __name__ == "__main__":
    main()
