import json
import glob
import os
import pandas as pd
import numpy as np
from scipy import stats

def analyze_phase_transitions(base_dir="results"):
    files = glob.glob(os.path.join(base_dir, "**", "results.json"), recursive=True)
    data = []

    for f in files:
        try:
            with open(f, 'r') as file:
                d = json.load(file)

            history = d.get('history', [])
            if not history or not d.get('grokked', False):
                continue

            config = d.get('config', {})
            collapse_level = config.get('collapse_level', 0.0)
            noise_fraction = config.get('noise_fraction', 0.0)
            severity = max(collapse_level, noise_fraction)

            # Find train acc convergence step (>0.99)
            train_conv_step = np.nan
            for h in history:
                if h.get('train_acc', 0) > 0.99:
                    train_conv_step = h.get('step')
                    break

            # Find test acc convergence step (>0.90)
            test_conv_step = np.nan
            for h in history:
                if h.get('test_acc', 0) > 0.90:
                    test_conv_step = h.get('step')
                    break

            grokking_gap = np.nan
            if pd.notna(train_conv_step) and pd.notna(test_conv_step) and test_conv_step >= train_conv_step:
                grokking_gap = test_conv_step - train_conv_step

            # Compute rate of change of fourier_concentration and embedding_rank
            fourier = [h.get('fourier_concentration', np.nan) for h in history]
            # Use embedding_rank to represent "weight rank evolution" as requested, fallback to weight_norm if missing
            rank = [h.get('embedding_rank', h.get('weight_norm', np.nan)) for h in history]
            steps = [h.get('step') for h in history]

            # Find onset of rapid change (max derivative)
            fourier_diff = np.diff(fourier)
            rank_diff = np.diff(rank)

            fourier_onset = steps[np.argmax(fourier_diff)] if len(fourier_diff) > 0 else np.nan
            rank_onset = steps[np.argmax(rank_diff)] if len(rank_diff) > 0 else np.nan

            condition = "pure" if severity == 0 else f"collapsed_{severity}"

            data.append({
                'file': f,
                'condition': condition,
                'severity': severity,
                'train_conv_step': train_conv_step,
                'test_conv_step': test_conv_step,
                'grokking_gap': grokking_gap,
                'fourier_onset_step': fourier_onset,
                'rank_onset_step': rank_onset
            })

        except Exception as e:
            print(f"Error processing {f}: {e}")

    df = pd.DataFrame(data)

    print("--- PHASE TRANSITION ANALYSIS ---")

    # Compare gap across severity
    pure = df[df['severity'] == 0].dropna(subset=['grokking_gap'])
    collapsed = df[df['severity'] > 0].dropna(subset=['grokking_gap'])

    print(f"Pure mean grokking gap: {pure['grokking_gap'].mean():.1f} steps (n={len(pure)})")
    if len(collapsed) > 0:
        print(f"Collapsed mean grokking gap: {collapsed['grokking_gap'].mean():.1f} steps (n={len(collapsed)})")
        t_stat, p_val = stats.ttest_ind(pure['grokking_gap'], collapsed['grokking_gap'], equal_var=False)
        print(f"T-test gap difference: t={t_stat:.4f}, p={p_val:.4e}")
    else:
        print("Not enough collapsed runs that successfully grokked to compare gap.")

    # Correlations with the gap
    df_valid = df.dropna(subset=['grokking_gap', 'fourier_onset_step', 'rank_onset_step'])
    if len(df_valid) > 2:
        r, p = stats.pearsonr(df_valid['grokking_gap'], df_valid['fourier_onset_step'])
        print(f"Correlation (Grokking Gap vs Fourier Onset Step): r={r:.4f}, p={p:.4e}")

        r, p = stats.pearsonr(df_valid['grokking_gap'], df_valid['rank_onset_step'])
        print(f"Correlation (Grokking Gap vs Weight Rank Onset Step): r={r:.4f}, p={p:.4e}")

    return df

if __name__ == "__main__":
    analyze_phase_transitions()
