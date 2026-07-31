import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class WeightAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.checkpoints = sorted(self.run_dir.glob("checkpoint_*.pt"),
                                  key=lambda x: int(x.stem.split('_')[1]))

        results_path = self.run_dir / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = None

    def detect_grokking(self, window_size: int = 5, threshold: float = 0.5) -> int:
        """
        Detect grokking moment (sudden jump in accuracy) algorithmically.
        """
        if not self.results or 'test_acc' not in self.results:
            return -1

        test_acc = np.array(self.results['test_acc'])
        steps = np.array(self.results.get('step', [i * self.results.get('eval_every', 100) for i in range(len(test_acc))]))

        if np.max(test_acc) < 0.9:
            return -1

        diffs = np.diff(test_acc)
        # Find points where accuracy jumps significantly
        jump_indices = np.where(diffs > threshold)[0]

        if len(jump_indices) > 0:
            return steps[jump_indices[0]]

        # If no single jump > threshold, find where it crosses 90%
        cross_90 = np.where(test_acc >= 0.9)[0]
        if len(cross_90) > 0:
            return steps[cross_90[0]]

        return -1

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            return state_dict['model_state_dict']
        return state_dict

    def extract_weight_stats(self) -> pd.DataFrame:
        """
        Track weight norm, rank, and singular value entropy over training.
        """
        records = []
        for ckpt in self.checkpoints:
            step = int(ckpt.stem.split('_')[1])
            state_dict = self.load_checkpoint(ckpt)

            stats = {'step': step}

            # Calculate metrics for specific layers
            layers_of_interest = {
                'token_embed': 'token_embed.weight',
                'output_head': 'output_head.weight',
                'q_proj': 'transformer.layers.0.self_attn.in_proj_weight' # approximate name, will handle properly
            }

            total_norm = 0.0

            for name, param in state_dict.items():
                # L2 norm
                norm = torch.norm(param, p=2).item()
                total_norm += norm ** 2

                # Check if it's a matrix (for SVD)
                if param.dim() == 2:
                    # Singular value decomposition
                    try:
                        s = torch.linalg.svdvals(param)
                        # Effective rank and entropy
                        s_norm = s / (s.sum() + 1e-10)
                        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
                        effective_rank = np.exp(entropy)

                        stats[f'{name}_norm'] = norm
                        stats[f'{name}_entropy'] = entropy
                        stats[f'{name}_rank'] = effective_rank
                    except Exception as e:
                        print(f"Warning: SVD failed for {name} at step {step}: {e}")

            stats['total_weight_norm'] = np.sqrt(total_norm)
            records.append(stats)

        return pd.DataFrame(records)

    def plot_weight_trajectories(self, df: pd.DataFrame, save_path: str = None):
        """Plot the trajectory of weight norms and effective ranks."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Total Weight Norm
        if 'total_weight_norm' in df.columns:
            sns.lineplot(data=df, x='step', y='total_weight_norm', ax=axes[0], label='Total Norm')

        # Plot Specific layer norms
        norm_cols = [c for c in df.columns if c.endswith('_norm') and c != 'total_weight_norm']
        for col in norm_cols:
            if 'token_embed' in col or 'output_head' in col: # Just a few key layers
                sns.lineplot(data=df, x='step', y=col, ax=axes[0], label=col, linestyle='--')

        axes[0].set_title('Weight Norm Evolution')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('L2 Norm')
        axes[0].legend()

        # Plot Effective Rank
        rank_cols = [c for c in df.columns if c.endswith('_rank')]
        for col in rank_cols:
            if 'token_embed' in col or 'output_head' in col:
                sns.lineplot(data=df, x='step', y=col, ax=axes[1], label=col)

        axes[1].set_title('Effective Rank Evolution')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Effective Rank')
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_all_runs(results_dir: str, output_dir: str):
    """
    Compare weight statistics between grokking and non-grokking runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_dir_path = Path(results_dir)

    all_summaries = []

    # Iterate through possible condition directories (e.g., pure, low_collapse, etc.)
    for run_dir in results_dir_path.glob("**/seed_*"):
        if not run_dir.is_dir() or not (run_dir / "results.json").exists():
            continue

        analyzer = WeightAnalyzer(str(run_dir))
        grok_step = analyzer.detect_grokking()
        has_grokked = grok_step != -1

        df = analyzer.extract_weight_stats()
        if df.empty:
            continue

        condition = run_dir.parent.name
        seed = run_dir.name

        # Save individual plots
        analyzer.plot_weight_trajectories(df, save_path=os.path.join(output_dir, f"{condition}_{seed}_trajectories.png"))

        final_norm = df.iloc[-1]['total_weight_norm'] if 'total_weight_norm' in df.columns else np.nan
        final_acc = analyzer.results.get('test_acc', [-1])[-1] if analyzer.results else -1

        summary = {
            'condition': condition,
            'seed': seed,
            'grok_step': grok_step,
            'has_grokked': has_grokked,
            'final_weight_norm': final_norm,
            'final_test_acc': final_acc
        }
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, "weight_summary.csv"), index=False)

        # Plot comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=summary_df, x='condition', y='final_weight_norm', hue='has_grokked')
        plt.title('Final Weight Norm by Condition and Grokking Status')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "norm_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    return summary_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="analysis/weights")
    args = parser.parse_args()

    analyze_all_runs(args.results_dir, args.output_dir)
