import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from analysis.analyze_results import load_all_results, build_dataframe

def set_paper_style():
    """Sets matplotlib rcParams for publication quality figures."""
    plt.style.use('default')
    sns.set_style("whitegrid")

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
    })

def plot_main_results(df: pd.DataFrame, output_dir: Path):
    """
    Plot accuracy curves across conditions over time (steps).
    """
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    available_conds = [c for c in conditions if c in df["condition"].unique()]
    colors = sns.color_palette("viridis", len(available_conds))

    plt.figure(figsize=(10, 6))

    for idx, cond in enumerate(available_conds):
        cond_df = df[df["condition"] == cond]
        if cond_df.empty:
            continue

        # We need to average histories across seeds
        all_histories = []
        for _, row in cond_df.iterrows():
            if "history" in row and isinstance(row["history"], list):
                all_histories.append(pd.DataFrame(row["history"]))

        if not all_histories:
            continue

        # Concatenate and group by step to get mean and std
        combined_hist = pd.concat(all_histories)
        grouped = combined_hist.groupby("step")["test_acc"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], label=cond, color=colors[idx])
        plt.fill_between(grouped["step"],
                         grouped["mean"] - grouped["std"],
                         grouped["mean"] + grouped["std"],
                         color=colors[idx], alpha=0.2)

    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Grokking threshold')

    plt.xlabel('Training Step')
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Model Collapse on Generalization')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "fig1_main_results.png")
    plt.savefig(output_dir / "fig1_main_results.pdf")
    plt.close()

def plot_grokking_timing(df: pd.DataFrame, output_dir: Path):
    """
    Plot grokking timing distribution.
    """
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    available_conds = [c for c in conditions if c in df["condition"].unique()]

    plt.figure(figsize=(8, 6))

    grokking_data = []
    labels = []

    for c in available_conds:
        steps = df[(df["condition"] == c) & (df["grokking_step"] > 0)]["grokking_step"].values
        if len(steps) > 0:
            grokking_data.append(steps)
            labels.append(c)

    if grokking_data:
        plt.boxplot(grokking_data, tick_labels=labels)
        plt.ylabel('Grokking Step')
        plt.title('Phase Transition Timing')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(output_dir / "fig2_grokking_timing.png")
        plt.savefig(output_dir / "fig2_grokking_timing.pdf")
        plt.close()

def plot_weight_fourier_dynamics(df: pd.DataFrame, output_dir: Path):
    """
    Plot weight norm evolution and Fourier concentration timeline.
    """
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    available_conds = [c for c in conditions if c in df["condition"].unique()]
    colors = sns.color_palette("magma", len(available_conds))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, cond in enumerate(available_conds):
        cond_df = df[df["condition"] == cond]
        if cond_df.empty:
            continue

        # We need to average histories across seeds
        all_histories = []
        for _, row in cond_df.iterrows():
            if "history" in row and isinstance(row["history"], list):
                all_histories.append(pd.DataFrame(row["history"]))

        if not all_histories:
            continue

        combined_hist = pd.concat(all_histories)

        # Plot Weight Norms
        if "weight_norm" in combined_hist.columns:
            grouped_wn = combined_hist.groupby("step")["weight_norm"].agg(["mean", "std"]).reset_index()
            axes[0].plot(grouped_wn["step"], grouped_wn["mean"], label=cond, color=colors[idx])
            axes[0].fill_between(grouped_wn["step"],
                             grouped_wn["mean"] - grouped_wn["std"],
                             grouped_wn["mean"] + grouped_wn["std"],
                             color=colors[idx], alpha=0.2)

        # Plot Fourier Concentration
        if "fourier_concentration" in combined_hist.columns:
            grouped_fc = combined_hist.groupby("step")["fourier_concentration"].agg(["mean", "std"]).reset_index()
            axes[1].plot(grouped_fc["step"], grouped_fc["mean"], label=cond, color=colors[idx])
            axes[1].fill_between(grouped_fc["step"],
                             grouped_fc["mean"] - grouped_fc["std"],
                             grouped_fc["mean"] + grouped_fc["std"],
                             color=colors[idx], alpha=0.2)

    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Weight Norm (L2)')
    axes[0].set_title('Weight Scale Growth')
    axes[0].legend()

    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Fourier Concentration')
    axes[1].set_title('Representation Periodicity')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_weight_fourier.png")
    plt.savefig(output_dir / "fig3_weight_fourier.pdf")
    plt.close()

if __name__ == "__main__":
    results_dir = Path("results")
    if results_dir.exists():
        set_paper_style()
        # Ensure we don't filter out history in load_all_results here
        # Actually load_all_results keeps history, build_dataframe drops it.
        # We need a build_dataframe that DOES NOT drop history for plotting.

        results_list = load_all_results(results_dir)
        df_with_history = pd.DataFrame(results_list) # Retains 'history'

        output_dir = Path("analysis/paper")
        output_dir.mkdir(exist_ok=True, parents=True)

        plot_main_results(df_with_history, output_dir)
        plot_grokking_timing(df_with_history, output_dir)
        plot_weight_fourier_dynamics(df_with_history, output_dir)
        print(f"Paper figures saved to {output_dir}")
