import json
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd
import itertools

def get_run_data(results_json_path):
    try:
        with open(results_json_path, 'r') as f:
            data = json.load(f)

        grokking_step = data.get('grokking_step')
        if not data.get('grokked', False):
            grokking_step = np.nan

        acc = data.get('final_test_acc', 0)

        history = data.get('history', [])
        norm_reduction = np.nan
        if history:
            norms = [entry.get('weight_norm', 0) for entry in history if 'weight_norm' in entry]
            if norms:
                peak = max(norms)
                final = norms[-1]
                if peak > 0:
                    norm_reduction = (peak - final) / peak

        return {
            'grokking_step': grokking_step,
            'final_test_acc': acc,
            'norm_reduction': norm_reduction,
        }
    except Exception:
        return None

def cohen_d(x, y):
    """Calculate Cohen's d for two groups."""
    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    dof = nx + ny - 2
    pool_var = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof
    if pool_var == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pool_var)

def permutation_test(x, y, n_permutations=10000):
    """Permutation test for difference in means."""
    x = np.array(x)
    y = np.array(y)

    # Remove NaNs
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    obs_diff = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y])

    count = 0
    np.random.seed(42)
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_x = pooled[:len(x)]
        perm_y = pooled[len(x):]
        if abs(np.mean(perm_x) - np.mean(perm_y)) >= abs(obs_diff):
            count += 1

    return count / n_permutations

def compute_stats(results_dir, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    multi_seed_dir = Path(results_dir) / 'multi_seed'

    data = {cond: {'grokking_steps': [], 'accs': [], 'norm_reductions': []} for cond in conditions}

    has_multi = False
    if multi_seed_dir.exists():
        for seed_dir in multi_seed_dir.iterdir():
            if not seed_dir.is_dir(): continue
            has_multi = True
            for cond in conditions:
                res_path = seed_dir / cond / 'results.json'
                if res_path.exists():
                    d = get_run_data(res_path)
                    if d:
                        data[cond]['grokking_steps'].append(d['grokking_step'])
                        data[cond]['accs'].append(d['final_test_acc'])
                        data[cond]['norm_reductions'].append(d['norm_reduction'])

    if not has_multi:
        print("Insufficient seeds for multi-seed statistical testing (found 1 or 0). Proceeding with single run extraction...")
        for cond in conditions:
            res_path = Path(results_dir) / cond / 'results.json'
            if res_path.exists():
                d = get_run_data(res_path)
                if d:
                    data[cond]['grokking_steps'].append(d['grokking_step'])
                    data[cond]['accs'].append(d['final_test_acc'])
                    data[cond]['norm_reductions'].append(d['norm_reduction'])

    # Generate summary table
    rows = []
    for cond in conditions:
        steps = np.array(data[cond]['grokking_steps'], dtype=float)
        accs = np.array(data[cond]['accs'], dtype=float)
        reds = np.array(data[cond]['norm_reductions'], dtype=float)

        valid_steps = steps[~np.isnan(steps)]

        rows.append({
            'Condition': cond,
            'N': len(accs),
            'Grok Fail Rate': np.isnan(steps).mean() if len(steps)>0 else np.nan,
            'Grokking Step (Mean ± SD)': f"{np.mean(valid_steps):.1f} ± {np.std(valid_steps):.1f}" if len(valid_steps)>0 else "NaN",
            'Final Acc (Mean ± SD)': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}" if len(accs)>0 else "NaN",
            'Norm Red (Mean ± SD)': f"{np.nanmean(reds):.3f} ± {np.nanstd(reds):.3f}" if len(reds)>0 else "NaN"
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path / 'summary_stats.csv', index=False)
    print("\nSummary Statistics:")
    print(df.to_string(index=False))

    # Statistical tests between pure and low_collapse if enough data
    pure_acc = data['pure']['accs']
    low_acc = data['low_collapse']['accs']

    pure_step = data['pure']['grokking_steps']
    low_step = data['low_collapse']['grokking_steps']

    print("\nStatistical Tests (Pure vs Low Collapse):")
    if len(pure_acc) >= 2 and len(low_acc) >= 2:
        # Acc
        p_acc = permutation_test(pure_acc, low_acc)
        d_acc = cohen_d(pure_acc, low_acc)
        print(f"Final Accuracy: p={p_acc:.4f}, Cohen's d={d_acc:.2f}")

        # Step
        p_step = permutation_test(pure_step, low_step)
        d_step = cohen_d(pure_step, low_step)
        print(f"Grokking Step: p={p_step:.4f}, Cohen's d={d_step:.2f}")

        with open(out_path / 'statistical_tests.txt', 'w') as f:
            f.write("Pure vs Low Collapse:\n")
            f.write(f"Final Accuracy: p={p_acc:.4f}, Cohen's d={d_acc:.2f}\n")
            f.write(f"Grokking Step: p={p_step:.4f}, Cohen's d={d_step:.2f}\n")
    else:
        print("Insufficient seeds for permutation testing (need >= 2 for both conditions).")
        with open(out_path / 'statistical_tests.txt', 'w') as f:
            f.write("Insufficient seeds for permutation testing (need >= 2 for both conditions).\n")

if __name__ == '__main__':
    compute_stats("results", "analysis/stats")
    print("\nStats analysis complete.")
