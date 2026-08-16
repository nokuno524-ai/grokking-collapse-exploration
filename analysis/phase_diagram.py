import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob

def get_run_data(results_json_path):
    try:
        with open(results_json_path, 'r') as f:
            data = json.load(f)

        grokking_step = data.get('grokking_step')
        if not data.get('grokked', False):
            grokking_step = np.nan

        acc = data.get('final_test_acc', 0)

        # Calculate norm reduction if history is available
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
            'collapse_severity': data.get('config', {}).get('collapse_severity', 0.0),
            'collapse_level': data.get('config', {}).get('collapse_level', 0.0)
        }
    except Exception as e:
        print(f"Error reading {results_json_path}: {e}")
        return None

def analyze_phase_diagram(results_dir, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Gather multi-seed data if available
    multi_seed_dir = Path(results_dir) / 'multi_seed'
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    data_by_cond = {cond: {'grokking_steps': [], 'accs': [], 'norm_reductions': [], 'severities': []} for cond in conditions}

    # Also gather single runs if multi-seed doesn't exist
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
                        data_by_cond[cond]['grokking_steps'].append(d['grokking_step'])
                        data_by_cond[cond]['accs'].append(d['final_test_acc'])
                        data_by_cond[cond]['norm_reductions'].append(d['norm_reduction'])
                        # pure has severity 0 theoretically, but config might say 0.5.
                        # We'll use our own severity mapping or the config one.
                        # Actually use the one from config, but for pure it's 0.0
                        sev = 0.0 if cond == "pure" else d['collapse_severity']
                        data_by_cond[cond]['severities'].append(sev)

    # If no multi-seed or we want to include root ones
    if not has_multi:
        for cond in conditions:
            res_path = Path(results_dir) / cond / 'results.json'
            if res_path.exists():
                d = get_run_data(res_path)
                if d:
                    data_by_cond[cond]['grokking_steps'].append(d['grokking_step'])
                    data_by_cond[cond]['accs'].append(d['final_test_acc'])
                    data_by_cond[cond]['norm_reductions'].append(d['norm_reduction'])
                    sev = 0.0 if cond == "pure" else d['collapse_severity']
                    data_by_cond[cond]['severities'].append(sev)

    # Summarize
    plot_data = []
    all_severities = []
    all_norm_reds = []
    all_accs = []

    for cond in conditions:
        steps = np.array(data_by_cond[cond]['grokking_steps'], dtype=float)
        accs = np.array(data_by_cond[cond]['accs'], dtype=float)
        reds = np.array(data_by_cond[cond]['norm_reductions'], dtype=float)
        sevs = np.array(data_by_cond[cond]['severities'], dtype=float)

        if len(steps) == 0:
            continue

        # Ignore NaNs for mean calculation of grokking step
        valid_steps = steps[~np.isnan(steps)]
        mean_step = np.mean(valid_steps) if len(valid_steps) > 0 else np.nan
        std_step = np.std(valid_steps) if len(valid_steps) > 0 else 0

        # Calculate grok failure rate
        fail_rate = np.isnan(steps).mean()

        mean_sev = np.nanmean(sevs)

        plot_data.append({
            'condition': cond,
            'severity': mean_sev,
            'mean_step': mean_step,
            'std_step': std_step,
            'fail_rate': fail_rate
        })

        all_severities.extend(sevs)
        all_norm_reds.extend(reds)
        all_accs.extend(accs)

    # Plot 1: Collapse Severity vs Grokking Step
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Extract points
    sevs = [d['severity'] for d in plot_data]
    steps = [d['mean_step'] for d in plot_data]
    stds = [d['std_step'] for d in plot_data]
    fails = [d['fail_rate'] for d in plot_data]
    labels = [d['condition'] for d in plot_data]

    # Plot successful groks
    for i, (sev, step, std, fail, label) in enumerate(zip(sevs, steps, stds, fails, labels)):
        if not np.isnan(step):
            ax1.errorbar(sev, step, yerr=std, fmt='o', markersize=10,
                         label=f'{label} ({1-fail:.0%} grok)', capsize=5)
        else:
            # Mark failures at the top of the plot
            ax1.plot(sev, 50000, 'rx', markersize=12, label=f'{label} (Failed)')

    ax1.set_xlabel('Collapse Severity')
    ax1.set_ylabel('Grokking Step (log scale)')
    ax1.set_yscale('log')
    if has_multi:
        ax1.set_title('Grokking Step vs Collapse Severity (Multi-seed)')
    else:
        ax1.set_title('Grokking Step vs Collapse Severity')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.savefig(out_path / 'phase_diagram_grokking.png', dpi=300)
    plt.savefig(out_path / 'phase_diagram_grokking.pdf', dpi=300)
    plt.close()

    # Plot 2: Norm reduction vs final accuracy
    fig, ax2 = plt.subplots(figsize=(8, 6))

    # Filter out NaNs
    valid = ~(np.isnan(all_norm_reds) | np.isnan(all_accs))
    x = np.array(all_norm_reds)[valid]
    y = np.array(all_accs)[valid]
    c = np.array(all_severities)[valid]

    scatter = ax2.scatter(x, y, c=c, cmap='viridis', s=100, alpha=0.8, edgecolors='k')
    plt.colorbar(scatter, label='Collapse Severity')

    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Grokking Threshold')

    ax2.set_xlabel('Weight Norm Reduction ((Peak - Final) / Peak)')
    ax2.set_ylabel('Final Test Accuracy')
    ax2.set_title('Norm Reduction vs Final Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path / 'phase_diagram_accuracy.png', dpi=300)
    plt.savefig(out_path / 'phase_diagram_accuracy.pdf', dpi=300)
    plt.close()

if __name__ == '__main__':
    analyze_phase_diagram("results", "analysis/phase")
    print("Phase diagram analysis complete.")
