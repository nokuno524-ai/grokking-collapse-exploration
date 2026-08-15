import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.utils import resample
from collections import defaultdict
import warnings

def load_run_data(filepath):
    """Loads a run log from a results.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(run_data, tau=0.99):
    """Extracts grok step, final val accuracy, weight norm trajectory, etc."""
    history = run_data.get('history', [])
    if not history:
        return {}

    test_accs = np.array([h['test_acc'] for h in history])
    steps = np.array([h['step'] for h in history])
    weight_norms = np.array([h['weight_norm'] for h in history])

    # Calculate grok step
    grok_indices = np.where(test_accs >= tau)[0]
    grok_step = steps[grok_indices[0]] if len(grok_indices) > 0 else None
    grok_success = (grok_step is not None)

    final_test_acc = run_data.get('final_test_acc', test_accs[-1])
    final_weight_norm = run_data.get('final_weight_norm', weight_norms[-1])

    # Continuous collapse severity (weight norm drop percentage)
    peak_wn = np.max(weight_norms)
    wn_drop_pct = (peak_wn - final_weight_norm) / peak_wn if peak_wn > 0 else 0.0

    attention_entropies = np.array([h.get('attention_entropy', np.nan) for h in history])

    config = run_data.get('config', {})
    noise_level = config.get('collapse_level', config.get('noise', 0.0))
    condition_name = config.get('condition_name', 'unknown')

    return {
        'grok_step': grok_step,
        'grok_success': grok_success,
        'final_test_acc': final_test_acc,
        'weight_norms': weight_norms,
        'steps': steps,
        'wn_drop_pct': wn_drop_pct,
        'attention_entropies': attention_entropies,
        'condition_name': condition_name,
        'noise_level': noise_level,
    }

def bootstrap_ci(data, func=np.mean, n_bootstraps=1000, ci=0.95):
    """Computes bootstrap confidence interval."""
    if len(data) == 0:
        return np.nan, np.nan
    bootstrapped_stats = []
    # Ensure reproducible
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        sample = resample(data, random_state=rng)
        bootstrapped_stats.append(func(sample))

    lower = np.percentile(bootstrapped_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped_stats, (1 + ci) / 2 * 100)
    return lower, upper

def categorical_analysis(runs):
    """Computes grok rate and delay (with CIs) for each categorical condition."""
    conditions = defaultdict(list)
    for r in runs:
        conditions[r['condition_name']].append(r)

    results = {}
    for cond, cond_runs in conditions.items():
        successes = [1.0 if r['grok_success'] else 0.0 for r in cond_runs]
        delays = [r['grok_step'] for r in cond_runs if r['grok_success'] and r['grok_step'] is not None]

        success_mean = np.mean(successes)
        success_ci = bootstrap_ci(successes, np.mean)

        if delays:
            delay_mean = np.mean(delays)
            delay_ci = bootstrap_ci(delays, np.mean)
        else:
            delay_mean = np.nan
            delay_ci = (np.nan, np.nan)

        results[cond] = {
            'n_runs': len(cond_runs),
            'success_rate': success_mean,
            'success_ci': success_ci,
            'mean_delay': delay_mean,
            'delay_ci': delay_ci
        }
    return results

def log_linear(x, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a * np.exp(b * x)

def logistic(x, L, k, x0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return L / (1 + np.exp(-k * (x - x0)))

def fit_delay_vs_severity(severities, delays):
    valid_mask = [d is not None and not np.isnan(d) for d in delays]
    valid_sevs = np.array(severities)[valid_mask]
    valid_dels = np.array(delays)[valid_mask]

    if len(valid_sevs) < 2:
        return None, None

    try:
        popt, pcov = curve_fit(log_linear, valid_sevs, valid_dels, p0=[1000, 1], maxfev=10000)
        return popt, np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        return None, None

def fit_success_vs_severity(severities, successes):
    sevs = np.array(severities)
    succs = np.array(successes, dtype=float)

    if len(sevs) < 3:
        return None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(logistic, sevs, succs, p0=[1, -10, np.mean(sevs)], bounds=([0, -np.inf, -np.inf], [1.1, np.inf, np.inf]), maxfev=10000)
            std_err = np.sqrt(np.diag(pcov)) if not np.isinf(pcov).any() else np.array([np.inf]*len(popt))
            return popt, std_err
    except (RuntimeError, ValueError):
        return None, None

def collect_all_runs(results_dir="results"):
    results_path = Path(results_dir)
    all_metrics = []

    for path in results_path.rglob("results.json"):
        if 'checkpoint' in path.parts:
            continue
        try:
            run_data = load_run_data(path)
            metrics = extract_metrics(run_data)
            if metrics:
                metrics['filepath'] = str(path)
                all_metrics.append(metrics)
        except Exception:
            pass

    return all_metrics
