"""
ANOVA-style decomposition for optimizer and regularization effects on grokking onset.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import scipy.stats as stats

def load_sweep_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load results from the optimizer sweep."""
    root = Path(results_dir)
    if not root.exists():
        return []

    results = []

    # Directory structure: root/opt_X/wd_Y/lr_Z/cond_W/seed_S/results.json
    for opt_dir in root.glob("opt_*"):
        opt = opt_dir.name.replace("opt_", "")

        for wd_dir in opt_dir.glob("wd_*"):
            wd = float(wd_dir.name.replace("wd_", ""))

            for lr_dir in wd_dir.glob("lr_*"):
                lr = float(lr_dir.name.replace("lr_", ""))

                for cond_dir in lr_dir.glob("cond_*"):
                    cond = cond_dir.name.replace("cond_", "")

                    for seed_dir in cond_dir.glob("seed_*"):
                        seed = int(seed_dir.name.replace("seed_", ""))

                        # Fix for nested directory structure issue during training
                        # It might be in seed_X/seed_X/results.json or seed_X/results.json
                        res_path = seed_dir / "results.json"
                        if not res_path.exists():
                            # Check if it was saved with condition_name as seed_dir
                            alt_path = seed_dir / f"seed_{seed}" / "results.json"
                            if alt_path.exists():
                                res_path = alt_path

                        try:
                            with open(res_path, 'r') as f:
                                data = json.load(f)

                            results.append({
                                'optimizer': opt,
                                'wd': wd,
                                'lr': lr,
                                'condition': cond,
                                'seed': seed,
                                'grokked': data.get('grokked', False),
                                'grokking_step': data.get('grokking_step'),
                                'final_test_acc': data.get('final_test_acc', 0.0)
                            })
                        except (json.JSONDecodeError, FileNotFoundError):
                            pass

    return results

def compute_interaction_stats(results: List[Dict[str, Any]],
                             max_steps: int = 50000) -> Dict[str, Any]:
    """
    Compute ANOVA-style statistics for interaction effects on grokking onset.
    Uses numpy for the decomposition and computes effect sizes.
    """
    if not results:
        return {}

    # Prepare data arrays
    # We treat runs that didn't grok as having grokking_step = max_steps
    steps = np.array([r['grokking_step'] if r['grokked'] and r['grokking_step'] is not None else max_steps
                     for r in results], dtype=float)

    optimizers = np.array([r['optimizer'] for r in results])
    wds = np.array([r['wd'] for r in results])
    conds = np.array([r['condition'] for r in results])

    # Get unique levels
    unique_opts = np.unique(optimizers)
    unique_wds = np.unique(wds)
    unique_conds = np.unique(conds)

    # Overall mean
    grand_mean = np.mean(steps)
    total_ss = np.sum((steps - grand_mean)**2)
    n_total = len(steps)

    stats_out = {
        'grand_mean': float(grand_mean),
        'n_total': n_total,
        'main_effects': {},
        'interactions': {}
    }

    # 1. Main effects

    # Optimizer main effect
    opt_ss = 0
    for opt in unique_opts:
        mask = (optimizers == opt)
        if np.any(mask):
            opt_mean = np.mean(steps[mask])
            opt_ss += np.sum(mask) * (opt_mean - grand_mean)**2
            stats_out['main_effects'][f'opt={opt}'] = float(opt_mean - grand_mean)

    # WD main effect
    wd_ss = 0
    for wd in unique_wds:
        mask = (wds == wd)
        if np.any(mask):
            wd_mean = np.mean(steps[mask])
            wd_ss += np.sum(mask) * (wd_mean - grand_mean)**2
            stats_out['main_effects'][f'wd={wd}'] = float(wd_mean - grand_mean)

    # Condition main effect
    cond_ss = 0
    for cond in unique_conds:
        mask = (conds == cond)
        if np.any(mask):
            cond_mean = np.mean(steps[mask])
            cond_ss += np.sum(mask) * (cond_mean - grand_mean)**2
            stats_out['main_effects'][f'cond={cond}'] = float(cond_mean - grand_mean)

    # 2. Two-way interactions

    # Opt x WD interaction
    opt_wd_ss = 0
    for opt in unique_opts:
        for wd in unique_wds:
            mask = (optimizers == opt) & (wds == wd)
            if np.any(mask):
                cell_mean = np.mean(steps[mask])
                opt_mean = np.mean(steps[optimizers == opt])
                wd_mean = np.mean(steps[wds == wd])
                interaction = cell_mean - opt_mean - wd_mean + grand_mean
                opt_wd_ss += np.sum(mask) * interaction**2
                stats_out['interactions'][f'opt={opt}:wd={wd}'] = float(interaction)

    # 3. Effect sizes (eta-squared)
    if total_ss > 0:
        stats_out['effect_sizes'] = {
            'eta2_optimizer': float(opt_ss / total_ss),
            'eta2_wd': float(wd_ss / total_ss),
            'eta2_condition': float(cond_ss / total_ss),
            'eta2_opt_wd': float(opt_wd_ss / total_ss)
        }

    # 4. Perform multiple comparisons (Holm-Bonferroni correction)
    # Let's compare pure vs collapsed for each optimizer setting
    p_values = []
    comparisons = []

    for opt in unique_opts:
        mask_pure = (optimizers == opt) & (conds == 'pure')
        mask_cond = (optimizers == opt) & (conds != 'pure')

        if np.any(mask_pure) and np.any(mask_cond):
            steps_pure = steps[mask_pure]
            steps_cond = steps[mask_cond]

            # Mann-Whitney U test (non-parametric, robust to non-normality/censoring)
            if len(steps_pure) >= 3 and len(steps_cond) >= 3:
                # We use two-sided since we don't know direction a priori
                stat, p = stats.mannwhitneyu(steps_pure, steps_cond, alternative='two-sided')
                comparisons.append(f"opt={opt}: pure vs collapsed")
                p_values.append(p)

    # Apply Holm-Bonferroni
    if p_values:
        p_values = np.array(p_values)
        sort_idx = np.argsort(p_values)
        sorted_p = p_values[sort_idx]
        sorted_comps = [comparisons[i] for i in sort_idx]

        m = len(p_values)
        alpha = 0.05
        corrected_p = np.zeros_like(p_values)
        significant = np.zeros_like(p_values, dtype=bool)

        for i in range(m):
            adj_alpha = alpha / (m - i)
            if sorted_p[i] <= adj_alpha:
                significant[i] = True
            else:
                break

        # Fill corrected p-values (step-down procedure)
        for i in range(m):
            # For reporting, calculate the adjusted p-value
            adj_p = sorted_p[i] * (m - i)
            # Ensure monotonicity
            if i > 0:
                adj_p = max(adj_p, corrected_p[i-1])
            corrected_p[i] = min(adj_p, 1.0)

        # Store results back in original order
        stats_out['multiple_comparisons'] = {}
        for i, idx in enumerate(sort_idx):
            stats_out['multiple_comparisons'][comparisons[idx]] = {
                'p_value': float(sorted_p[i]),
                'holm_corrected_p': float(corrected_p[i]),
                'significant': bool(significant[i])
            }

    return stats_out

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/opt_sweep"
    results = load_sweep_results(results_dir)
    stats = compute_interaction_stats(results)
    print(json.dumps(stats, indent=2))
