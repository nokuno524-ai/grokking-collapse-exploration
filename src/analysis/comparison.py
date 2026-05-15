"""
Comparison framework for analyzing grokking dynamics and collapse effects.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ComparisonFramework:
    """Framework for comparing grokking dynamics and detecting phase transitions."""

    @staticmethod
    def measure_time_to_grokking(history: List[Dict[str, Any]], threshold: float = 0.9, consecutive_steps: int = 50) -> Optional[int]:
        """
        Measure the step at which validation/test accuracy exceeds the threshold
        and remains above it for at least N consecutive evaluation steps.
        """
        if not history:
            return None

        acc_above_threshold_count = 0
        first_step_above = None

        # Sort history by step just in case
        sorted_history = sorted(history, key=lambda x: x.get("step", 0))

        for entry in sorted_history:
            # Prefer test_acc, fallback to train_acc if not available
            acc = entry.get("test_acc")
            if acc is None:
                acc = entry.get("train_acc", 0.0)

            if acc >= threshold:
                if acc_above_threshold_count == 0:
                    first_step_above = entry.get("step")
                acc_above_threshold_count += 1

                # if we treat consecutive_steps as 'count of eval steps' instead of 'training steps'
                if acc_above_threshold_count >= consecutive_steps:
                    return first_step_above
            else:
                acc_above_threshold_count = 0
                first_step_above = None

        return None

    @staticmethod
    def compute_auc(history: List[Dict[str, Any]], metric: str = "test_acc") -> float:
        """Compute the area under the curve for a specific metric over training steps."""
        if not history:
            return 0.0

        sorted_history = sorted(history, key=lambda x: x.get("step", 0))
        steps = [entry.get("step", 0) for entry in sorted_history]
        values = [entry.get(metric, 0.0) for entry in sorted_history]

        if len(steps) < 2:
            return 0.0

        return float(integrate.trapezoid(y=values, x=steps))

    @staticmethod
    def detect_phase_transitions(history: List[Dict[str, Any]], metric: str = "test_acc", threshold_derivative: float = 0.01) -> List[int]:
        """
        Detect sudden jumps (phase transitions) in a metric by calculating
        the discrete derivative (change per step).
        """
        if len(history) < 2:
            return []

        sorted_history = sorted(history, key=lambda x: x.get("step", 0))
        steps = [entry.get("step", 0) for entry in sorted_history]
        values = [entry.get(metric, 0.0) for entry in sorted_history]

        transitions = []
        for i in range(1, len(steps)):
            delta_val = values[i] - values[i-1]
            delta_step = steps[i] - steps[i-1]

            if delta_step > 0:
                derivative = delta_val / delta_step
                if derivative > threshold_derivative:
                    transitions.append(steps[i])

        return transitions

    @staticmethod
    def correlate_collapse_with_grokking(results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Correlate collapse metrics (like weight norm reduction) with grokking success.
        Expects a DataFrame containing 'collapse_severity', 'final_weight_norm', 'grokking_step' etc.
        """
        if results_df.empty:
            return {}

        correlations = {}

        if 'collapse_severity' in results_df.columns and 'grokking_step' in results_df.columns:
            # For correlation, we might want to fill NaNs in grokking_step with max_step or drop them
            df_clean = results_df.dropna(subset=['collapse_severity', 'grokking_step'])
            if len(df_clean) > 1:
                corr = df_clean['collapse_severity'].corr(df_clean['grokking_step'])
                correlations['severity_vs_grokking_step'] = corr

        if 'final_weight_norm' in results_df.columns and 'grokking_step' in results_df.columns:
            df_clean = results_df.dropna(subset=['final_weight_norm', 'grokking_step'])
            if len(df_clean) > 1:
                corr = df_clean['final_weight_norm'].corr(df_clean['grokking_step'])
                correlations['weight_norm_vs_grokking_step'] = corr

        return correlations

    @staticmethod
    def generate_comparison_plots(results: Dict[str, List[Dict[str, Any]]], output_path: str = "comparison_plot.png") -> None:
        """
        Generate comparison plots across collapse levels including:
        - Accuracy curves overlay
        - Grokking step bar charts
        - Final test accuracy violin plots
        """
        if not results:
            logger.warning("No data provided to generate plots.")
            return

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        ax_acc = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_violin = fig.add_subplot(gs[1, :])

        conditions = list(results.keys())
        colors = sns.color_palette("husl", len(conditions))

        # 1. Accuracy curves overlay (average over seeds)
        for i, condition in enumerate(conditions):
            runs = results[condition]
            if not runs:
                continue

            # Assume all runs have same steps
            first_run = runs[0].get("history", [])
            steps = [e.get("step") for e in first_run]

            acc_matrix = []
            for run in runs:
                history = run.get("history", [])
                if history:
                    # Align by step if possible, here we assume aligned
                    accs = [e.get("test_acc", 0) for e in history]
                    if len(accs) == len(steps):
                        acc_matrix.append(accs)

            if acc_matrix:
                mean_acc = np.mean(acc_matrix, axis=0)
                std_acc = np.std(acc_matrix, axis=0)
                ax_acc.plot(steps, mean_acc, label=condition, color=colors[i])
                ax_acc.fill_between(steps, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color=colors[i])

        ax_acc.set_title("Test Accuracy Trajectories")
        ax_acc.set_xlabel("Steps")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        # 2. Grokking step bar chart
        grokking_steps = []
        cond_labels = []
        for condition, runs in results.items():
            steps = [r.get("grokking_step") for r in runs if r.get("grokking_step") is not None]
            if steps:
                grokking_steps.append(np.mean(steps))
            else:
                grokking_steps.append(0)  # Represent no grokking
            cond_labels.append(condition)

        ax_bar.bar(cond_labels, grokking_steps, color=colors)
        ax_bar.set_title("Average Grokking Step")
        ax_bar.set_ylabel("Step")
        plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=45)

        # 3. Final test accuracy violin plots
        plot_data = []
        for condition in conditions:
            accs = [r.get("final_test_acc", 0) for r in results[condition]]
            for acc in accs:
                plot_data.append({"Condition": condition, "Accuracy": acc})

        df_plot = pd.DataFrame(plot_data)
        if not df_plot.empty:
            sns.violinplot(data=df_plot, x="Condition", y="Accuracy", ax=ax_violin, palette="husl")
            ax_violin.set_title("Final Test Accuracy Distribution")
            ax_violin.set_ylim(-0.1, 1.1)
            plt.setp(ax_violin.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
