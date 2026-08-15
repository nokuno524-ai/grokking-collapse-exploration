import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc

import analysis.linkage as link
import analysis.predictors as pred

plt.style.use('seaborn-v0_8-colorblind')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'text.usetex': False,
})

def plot_severity_vs_delay(runs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    grokked_runs = [r for r in runs if r['grok_success'] and r['grok_step'] is not None]

    if not grokked_runs:
        print("No successful grokking runs found for plotting.")
        return

    severities = [r['wn_drop_pct'] for r in grokked_runs]
    delays = [r['grok_step'] for r in grokked_runs]

    plt.figure()
    plt.scatter(severities, delays, alpha=0.6, label='Individual Runs')

    popt, std_err = link.fit_delay_vs_severity(severities, delays)
    if popt is not None:
        x_vals = np.linspace(min(severities), max(severities), 100)
        y_vals = link.log_linear(x_vals, *popt)

        # Plot curve with CIs
        y_lower = link.log_linear(x_vals, popt[0] - 1.96*std_err[0], popt[1] - 1.96*std_err[1])
        y_upper = link.log_linear(x_vals, popt[0] + 1.96*std_err[0], popt[1] + 1.96*std_err[1])

        # Correctly sort bounds based on correlation for shading
        y_min_shade = np.minimum(y_lower, y_upper)
        y_max_shade = np.maximum(y_lower, y_upper)

        plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Log-Linear Fit')
        plt.fill_between(x_vals, y_min_shade, y_max_shade, color='red', alpha=0.2, label='95% CI')

    plt.xlabel('Collapse Severity (Weight Norm Drop %)')
    plt.ylabel('Grokking Delay (Steps)')
    plt.title('Grokking Delay vs Collapse Severity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'severity_vs_delay.png', dpi=300)
    plt.close()

def plot_weight_norm_trajectories(runs, out_dir):
    out_dir = Path(out_dir)

    plt.figure()

    success_plotted = False
    fail_plotted = False

    for r in runs:
        steps = r['steps']
        wn = r['weight_norms']

        if r['grok_success']:
            color = 'blue'
            alpha = 0.2
            label = 'Grokked' if not success_plotted else ""
            success_plotted = True
        else:
            color = 'red'
            alpha = 0.2
            label = 'Failed to Grok' if not fail_plotted else ""
            fail_plotted = True

        plt.plot(steps, wn, color=color, alpha=alpha, label=label)

    plt.xlabel('Training Steps')
    plt.ylabel('Total Weight Norm')
    plt.title('Weight Norm Trajectories by Outcome')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if '' in by_label:
        del by_label['']
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(out_dir / 'weight_norm_trajectories.png', dpi=300)
    plt.close()

def plot_predictor_roc(y_true, y_scores, auroc, ci, out_dir):
    out_dir = Path(out_dir)

    if y_true is None or y_scores is None:
        print("Invalid data for ROC plot.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f} [{ci[0]:.2f}-{ci[1]:.2f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Early-Warning Predictor ROC (Pre-Grok Window)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / 'predictor_roc.png', dpi=300)
    plt.close()

def main():
    runs = link.collect_all_runs()
    print(f"Collected {len(runs)} runs.")

    plot_severity_vs_delay(runs, 'analysis/')
    plot_weight_norm_trajectories(runs, 'analysis/')

    early_metrics_list = []
    for r in runs:
        run_data = link.load_run_data(r['filepath'])
        early = pred.compute_early_warning_metrics(run_data, pre_grok_steps=1000)
        early_metrics_list.append(early)

    clf, auroc, ci, y_true, y_scores = pred.train_predictor_and_evaluate(runs, early_metrics_list)

    if y_true is not None:
        plot_predictor_roc(y_true, y_scores, auroc, ci, 'analysis/')
        print(f"Plots generated in analysis/ directory. AUROC: {auroc:.3f} 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    else:
        print("Failed to compute AUROC.")

    print("\nCategorical Analysis:")
    cat_res = link.categorical_analysis(runs)
    for k, v in cat_res.items():
        print(f"Condition: {k}")
        print(f"  Runs: {v['n_runs']}")
        print(f"  Success Rate: {v['success_rate']:.3f} CI: [{v['success_ci'][0]:.3f}, {v['success_ci'][1]:.3f}]")
        print(f"  Mean Delay: {v['mean_delay']:.1f} CI: [{v['delay_ci'][0]:.1f}, {v['delay_ci'][1]:.1f}]")

if __name__ == "__main__":
    main()
