import numpy as np
import scipy.stats

def correlate_early_fourier_with_grokking(history_list: list, early_steps_cutoff: int = 1000) -> dict:
    """
    Calculate the Pearson correlation between early Fourier concentration
    and the final grokking step/success.

    Args:
        history_list (list): A list of training trajectory histories.
                             Each history is a list of dictionaries containing metrics per step.
                             There should be a 'grokking_step' key, or it's inferred from test_acc.
        early_steps_cutoff (int): The step up to which we average the Fourier concentration.

    Returns:
        dict: Correlation statistics.
    """
    early_fourier_means = []
    grokking_steps = []
    grokking_success = []

    for history in history_list:
        if not history:
            continue

        # Extract early Fourier concentration
        early_fcs = [e.get("fourier_concentration", 0.0) for e in history if e.get("step", 0) <= early_steps_cutoff]
        if not early_fcs:
            continue

        avg_early_fc = np.mean(early_fcs)

        # Determine grokking step and success
        grok_step = None
        for entry in history:
            if entry.get("test_acc", 0) >= 0.95:
                grok_step = entry.get("step")
                break

        early_fourier_means.append(avg_early_fc)
        grokking_success.append(1.0 if grok_step is not None else 0.0)

        # If no grokking, we represent it as NaN for the grokking step correlation
        grokking_steps.append(float(grok_step) if grok_step is not None else np.nan)

    if len(early_fourier_means) < 2:
        return {
            'grokking_step_corr': np.nan,
            'grokking_step_pval': np.nan,
            'grokking_success_corr': np.nan,
            'grokking_success_pval': np.nan,
            'n_samples': len(early_fourier_means)
        }

    early_fourier_means = np.array(early_fourier_means)
    grokking_steps = np.array(grokking_steps)
    grokking_success = np.array(grokking_success)

    # Calculate correlation with grokking success
    # If all success values are the same (e.g. all 0 or all 1), variance is 0, correlation is undefined
    if np.var(grokking_success) == 0:
        success_corr, success_pval = np.nan, np.nan
    else:
        success_corr, success_pval = scipy.stats.pearsonr(early_fourier_means, grokking_success)

    # Calculate correlation with grokking step (only for successful runs)
    valid_idx = ~np.isnan(grokking_steps)
    valid_fourier = early_fourier_means[valid_idx]
    valid_steps = grokking_steps[valid_idx]

    if len(valid_steps) < 2 or np.var(valid_steps) == 0:
        step_corr, step_pval = np.nan, np.nan
    else:
        step_corr, step_pval = scipy.stats.pearsonr(valid_fourier, valid_steps)

    return {
        'grokking_step_corr': float(step_corr),
        'grokking_step_pval': float(step_pval),
        'grokking_success_corr': float(success_corr),
        'grokking_success_pval': float(success_pval),
        'n_samples': len(early_fourier_means),
        'n_grokked_samples': np.sum(valid_idx)
    }
