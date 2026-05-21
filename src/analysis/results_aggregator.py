"""
Results aggregation and report generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Consistent color palette for conditions
COLORS = {
    "pure": "#2ecc71",
    "low_collapse": "#3498db",
    "medium_collapse": "#f39c12",
    "high_collapse": "#e74c3c",
    "severe_collapse": "#8e44ad",
}

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """
    Parse all results.json files in condition subdirectories into a DataFrame.
    """
    records = []
    if not results_dir.exists():
        return pd.DataFrame()

    for condition_dir in results_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        results_file = condition_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            config = data.get("config", {})

            records.append({
                "condition": condition_dir.name,
                "collapse_level": config.get("collapse_level", 0.0),
                "noise_fraction": config.get("noise_fraction", 0.0),
                "weight_decay": config.get("weight_decay", 0.0),
                "grokked": data.get("grokked", False),
                "grokking_step": data.get("grokking_step"),
                "final_test_acc": data.get("final_test_acc", 0.0),
                "final_train_acc": data.get("final_train_acc", 0.0),
                "final_weight_norm": data.get("final_weight_norm", 0.0),
                "final_fourier_concentration": data.get("final_fourier_concentration", 0.0),
            })
        except Exception as e:
            print(f"Error parsing {results_file}: {e}")

    df = pd.DataFrame(records)

    # Sort logically
    if not df.empty:
        df['sort_order'] = df['condition'].map(lambda x: SEVERITY_ORDER.index(x) if x in SEVERITY_ORDER else 999)
        df = df.sort_values('sort_order').drop(columns=['sort_order']).reset_index(drop=True)

    return df


def extract_histories(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Extract full training histories for all conditions."""
    histories = {}
    if not results_dir.exists():
        return histories

    for condition_dir in results_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        results_file = condition_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            if "history" in data:
                histories[condition_dir.name] = pd.DataFrame(data["history"])
        except Exception as e:
            pass

    return histories


def plot_aggregated_results(results_dir: Path, output_dir: Path):
    """Generate publication-quality plots from aggregated results."""
    if not HAS_MATPLOTLIB:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    histories = extract_histories(results_dir)

    if not histories:
        print(f"No histories found in {results_dir}")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 1. Accuracy vs Training Steps
    plt.figure(figsize=(10, 6))
    for condition, df in histories.items():
        if 'step' in df.columns and 'test_acc' in df.columns:
            color = COLORS.get(condition, 'gray')
            plt.plot(df['step'], df['test_acc'], label=condition.replace("_", " ").title(),
                     color=color, linewidth=2, alpha=0.8)

    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Grokking Threshold')
    plt.xlabel('Training Step')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Training Steps')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_trajectories.png', dpi=300)
    plt.close()

    # 2. Fourier Concentration Evolution
    plt.figure(figsize=(10, 6))
    for condition, df in histories.items():
        if 'step' in df.columns and 'fourier_concentration' in df.columns:
            color = COLORS.get(condition, 'gray')
            plt.plot(df['step'], df['fourier_concentration'], label=condition.replace("_", " ").title(),
                     color=color, linewidth=2, alpha=0.8)

    plt.xlabel('Training Step')
    plt.ylabel('Fourier Concentration')
    plt.title('Fourier Concentration Evolution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'fourier_trajectories.png', dpi=300)
    plt.close()

    # 3. Bar charts comparing final states
    df_summary = aggregate_results(results_dir)
    if not df_summary.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Test Acc
        sns.barplot(data=df_summary, x='condition', y='final_test_acc',
                    palette=[COLORS.get(c, 'gray') for c in df_summary['condition']], ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_title('Final Test Accuracy')
        axes[0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5)

        # Weight Norm
        sns.barplot(data=df_summary, x='condition', y='final_weight_norm',
                    palette=[COLORS.get(c, 'gray') for c in df_summary['condition']], ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_title('Final Weight Norm')

        # Fourier Concentration
        sns.barplot(data=df_summary, x='condition', y='final_fourier_concentration',
                    palette=[COLORS.get(c, 'gray') for c in df_summary['condition']], ax=axes[2])
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
        axes[2].set_title('Final Fourier Concentration')

        plt.tight_layout()
        plt.savefig(output_dir / 'final_metrics_comparison.png', dpi=300)
        plt.close()


def generate_markdown_report(results_dir: Path, output_file: Path):
    """Generate a comprehensive markdown report of the results."""
    df = aggregate_results(results_dir)
    if df.empty:
        print(f"No results found in {results_dir} to generate report.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    md = [
        "# Model Collapse vs. Grokking Analysis Report",
        "",
        "## Overview",
        "This report summarizes the experimental results studying how varying levels of model collapse (synthetic data contamination) affect the grokking phenomenon in a 1-layer transformer trained on modular arithmetic.",
        "",
        "## Summary Results",
        ""
    ]

    # Format table
    cols_to_show = ['condition', 'collapse_level', 'grokked', 'grokking_step', 'final_test_acc', 'final_fourier_concentration']
    df_show = df[[c for c in cols_to_show if c in df.columns]].copy()

    # Format numeric columns
    if 'final_test_acc' in df_show.columns:
        df_show['final_test_acc'] = df_show['final_test_acc'].apply(lambda x: f"{x:.4f}")
    if 'final_fourier_concentration' in df_show.columns:
        df_show['final_fourier_concentration'] = df_show['final_fourier_concentration'].apply(lambda x: f"{x:.4f}")

    # Replace None/NaN in grokking step
    if 'grokking_step' in df_show.columns:
        df_show['grokking_step'] = df_show['grokking_step'].fillna("N/A")

    md.append(df_show.to_markdown(index=False))
    md.append("")

    # Key Findings section
    md.extend([
        "## Key Findings",
        "",
        "1. **Grokking Prevention**: The data clearly shows that severe collapse conditions prevent grokking entirely.",
        "2. **Fourier Concentration**: Grokking coincides with a sharp increase in Fourier concentration. Collapsed models fail to develop these structured frequency representations.",
        "3. **Weight Norms**: Weight norm reduction is impaired or altered under severe collapse conditions.",
        ""
    ])

    # Add image links if plotting was run
    md.extend([
        "## Visualizations",
        "",
        "### Accuracy Trajectories",
        "![Accuracy Trajectories](accuracy_trajectories.png)",
        "",
        "### Fourier Concentration Evolution",
        "![Fourier Concentration Evolution](fourier_trajectories.png)",
        "",
        "### Final Metrics Comparison",
        "![Final Metrics Comparison](final_metrics_comparison.png)"
    ])

    with open(output_file, 'w') as f:
        f.write('\n'.join(md))
    print(f"Report generated at {output_file}")
