import os
import json
import numpy as np
from scipy import stats
import statsmodels.api as sm

def load_metrics(metrics_path="analysis/attention_metrics.json"):
    """Load the computed attention metrics."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_accuracies(results_dir="results"):
    """
    Load test accuracies from the main results directory.
    Returns a dictionary: condition -> list of dicts {'step': step, 'test_acc': acc}
    """
    accuracies = {}
    conditions = ['pure', 'low_collapse', 'medium_collapse', 'severe_collapse', 'high_collapse']

    for condition in conditions:
        cond_dir = os.path.join(results_dir, condition)
        if not os.path.exists(cond_dir):
            continue

        seed_dirs = [d for d in os.listdir(cond_dir) if d.startswith('seed_')]
        if not seed_dirs:
            continue

        # Use first seed for matching with attention metrics
        results_file = os.path.join(cond_dir, seed_dirs[0], 'results.json')
        if not os.path.exists(results_file):
            continue

        with open(results_file, 'r') as f:
            data = json.load(f)
            if 'test_acc' in data:
                accuracies[condition] = [
                    {'step': k, 'test_acc': v}
                    for k, v in sorted([(int(step), acc) for step, acc in data['test_acc'].items()])
                ]

    return accuracies

def correlate_entropy_with_accuracy(metrics, accuracies, output_path="analysis/statistics.json"):
    """
    Compute Pearson and Spearman correlation between attention entropy and test accuracy
    for each condition.
    """
    results = {}

    for condition in metrics:
        if condition not in accuracies:
            continue

        # Match steps
        entropy_dict = {d['step']: d['entropy_total'] for d in metrics[condition]}
        acc_dict = {d['step']: d['test_acc'] for d in accuracies[condition]}

        common_steps = sorted(list(set(entropy_dict.keys()) & set(acc_dict.keys())))

        if len(common_steps) < 5:
            continue

        x = [entropy_dict[s] for s in common_steps]
        y = [acc_dict[s] for s in common_steps]

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        results[condition] = {
            'pearson': {'r': pearson_r, 'p_value': pearson_p},
            'spearman': {'r': spearman_r, 'p_value': spearman_p},
            'n_samples': len(common_steps)
        }

    return results

def test_significance_between_conditions(metrics):
    """
    Test if the final attention entropy is significantly different between
    pure (grokking) and collapsed conditions.
    Note: Ideally this would use multiple seeds. Since our metrics dict might only
    have one seed for deep attention tracking, we compare the last N steps in the plateau.
    """
    results = {}

    if 'pure' not in metrics or len(metrics['pure']) < 10:
        return results

    # Use the last 10 steps of the pure run as the baseline "plateau"
    pure_plateau = [d['entropy_total'] for d in metrics['pure'][-10:]]

    for condition in metrics:
        if condition == 'pure' or len(metrics[condition]) < 10:
            continue

        cond_plateau = [d['entropy_total'] for d in metrics[condition][-10:]]

        # Welch's t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(pure_plateau, cond_plateau, equal_var=False)

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p_val = stats.mannwhitneyu(pure_plateau, cond_plateau, alternative='two-sided')
        except ValueError:
            u_stat, u_p_val = None, None

        results[f"pure_vs_{condition}"] = {
            'welch_t': {'statistic': float(t_stat), 'p_value': float(p_val)},
            'mann_whitney': {'statistic': float(u_stat) if u_stat is not None else None,
                             'p_value': float(u_p_val) if u_p_val is not None else None},
            'pure_mean': float(np.mean(pure_plateau)),
            'cond_mean': float(np.mean(cond_plateau)),
            'difference': float(np.mean(pure_plateau) - np.mean(cond_plateau))
        }

    return results

def predict_grokking_from_early_entropy(metrics, accuracies):
    """
    Logistic regression: Can early attention entropy predict if the model will grok?
    We define "grokking" as reaching > 90% test accuracy at the final step.
    We define "early" as step 1000.
    """
    results = {}

    # We need to build a dataset across conditions
    X = []
    y = []

    for condition in metrics:
        if condition not in accuracies:
            continue

        # Check if it grokked (final accuracy > 0.9)
        final_acc = accuracies[condition][-1]['test_acc']
        grokked = 1 if final_acc > 0.9 else 0

        # Get early entropy (closest to step 1000)
        early_step = 1000
        closest = min(metrics[condition], key=lambda d: abs(d['step'] - early_step))

        # Only use if we actually have an early step (<= 5000)
        if closest['step'] <= 5000:
            X.append([closest['entropy_total']])
            y.append(grokked)

    if len(X) < 3 or len(set(y)) < 2:
        return {"error": "Not enough data or variance (need both grokked and not-grokked conditions) to fit regression."}

    try:
        X = sm.add_constant(X)
        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        results = {
            'pseudo_r2': float(result.prsquared) if not np.isnan(result.prsquared) else None,
            'p_value': float(result.llr_pvalue) if not np.isnan(result.llr_pvalue) else None,
            'coefficient': float(result.params[1]),
            'n_samples': len(y)
        }
    except Exception as e:
        results = {"error": str(e)}

    return results

def run_all_statistics(metrics_path="analysis/attention_metrics.json", output_path="analysis/statistics_results.json"):
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    metrics = load_metrics(metrics_path)
    accuracies = load_accuracies()

    results = {
        'correlations': correlate_entropy_with_accuracy(metrics, accuracies),
        'significance': test_significance_between_conditions(metrics),
        'regression': predict_grokking_from_early_entropy(metrics, accuracies)
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Statistics saved to {output_path}")

if __name__ == "__main__":
    run_all_statistics()
