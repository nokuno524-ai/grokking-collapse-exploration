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

# Add src to path so we can import model
import sys
sys.path.append('.')
from src.model import ModularArithmeticTransformer

class AttentionAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.checkpoints = sorted(self.run_dir.glob("checkpoint_*.pt"),
                                  key=lambda x: int(x.stem.split('_')[1]))

        # We need a model instance to extract cleanly
        self.model = ModularArithmeticTransformer()

    def load_checkpoint(self, checkpoint_path: Path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def extract_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manually extract attention weights since nn.TransformerEncoderLayer
        doesn't return them directly.
        """
        batch_size = x.shape[0]

        # 1. Forward pass up to attention
        seq_len = x.size(1)
        tok = self.model.token_embed(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.model.pos_embed(positions)
        h = tok + pos

        # 2. Extract Q, K, V manually for the first layer
        attn_layer = self.model.transformer.layers[0].self_attn

        # Project inputs
        qkv = torch.nn.functional.linear(
            h,
            attn_layer.in_proj_weight,
            attn_layer.in_proj_bias
        )

        # Split Q, K, V
        d_model = self.model.d_model
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        num_heads = self.model.n_heads
        head_dim = d_model // num_heads

        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        return attn_weights.detach()

    def compute_metrics(self, attn_weights: torch.Tensor) -> dict:
        """Compute entropy and concentration metrics."""
        # attn_weights shape: (batch, n_heads, seq_len, seq_len)

        # 1. Entropy per head
        # p * log(p) over the last dimension (key positions)
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
        mean_entropy = entropy.mean(dim=(0, 2)).numpy() # Average over batch and queries

        # 2. Concentration (max weight per head)
        max_weight = attn_weights.max(dim=-1)[0]
        mean_concentration = max_weight.mean(dim=(0, 2)).numpy()

        return {
            'entropy': mean_entropy,
            'concentration': mean_concentration
        }

    def analyze_training_evolution(self, test_data: torch.Tensor):
        """Analyze attention patterns across all checkpoints."""
        records = []

        for ckpt in self.checkpoints:
            step = int(ckpt.stem.split('_')[1])
            self.load_checkpoint(ckpt)

            weights = self.extract_attention(test_data)
            metrics = self.compute_metrics(weights)

            for head_idx in range(self.model.n_heads):
                records.append({
                    'step': step,
                    'head': head_idx,
                    'entropy': metrics['entropy'][head_idx],
                    'concentration': metrics['concentration'][head_idx]
                })

        return pd.DataFrame(records)

    def plot_attention_heatmaps(self, test_data: torch.Tensor, step: int, output_dir: str):
        """Plot attention heatmaps for a specific step."""
        # Find the checkpoint closest to this step
        target_ckpt = None
        for ckpt in self.checkpoints:
            ckpt_step = int(ckpt.stem.split('_')[1])
            if ckpt_step == step:
                target_ckpt = ckpt
                break

        if target_ckpt is None:
            # Get latest if exact step not found
            target_ckpt = self.checkpoints[-1]
            step = int(target_ckpt.stem.split('_')[1])

        self.load_checkpoint(target_ckpt)
        weights = self.extract_attention(test_data)

        # Average over batch
        mean_weights = weights.mean(dim=0).numpy() # (n_heads, seq_len, seq_len)

        fig, axes = plt.subplots(1, self.model.n_heads, figsize=(4 * self.model.n_heads, 4))
        if self.model.n_heads == 1:
            axes = [axes]

        for h in range(self.model.n_heads):
            sns.heatmap(mean_weights[h], ax=axes[h], vmin=0, vmax=1, cmap='viridis',
                        xticklabels=['a', 'b'], yticklabels=['a', 'b'])
            axes[h].set_title(f'Head {h}')

        plt.suptitle(f'Attention Weights at Step {step}')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'attention_heatmap_step_{step}.png'), dpi=300)
        plt.close()


def compare_conditions(results_dir: str, output_dir: str):
    """Compare attention patterns between collapse conditions."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate some test data to probe attention
    from src.data import generate_modular_arithmetic, DatasetConfig
    _, _, test_inputs, _ = generate_modular_arithmetic(DatasetConfig(prime=59))
    test_batch = test_inputs[:128] # Use a batch for stable statistics

    all_metrics = []

    results_dir_path = Path(results_dir)
    for run_dir in results_dir_path.glob("**/seed_*"):
        if not run_dir.is_dir() or not list(run_dir.glob("checkpoint_*.pt")):
            continue

        condition = run_dir.parent.name
        seed = run_dir.name

        analyzer = AttentionAnalyzer(str(run_dir))

        # Get metrics over time
        df = analyzer.analyze_training_evolution(test_batch)
        df['condition'] = condition
        df['seed'] = seed
        all_metrics.append(df)

        # Plot heatmaps at key milestones (assuming 50k max steps)
        # Pre-grok (1k), Grokking region (10k), Post-grok (50k)
        for milestone in [1000, 10000, 50000]:
            analyzer.plot_attention_heatmaps(
                test_batch,
                milestone,
                os.path.join(output_dir, f"{condition}_{seed}")
            )

    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # Plot entropy evolution across conditions
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=combined_df, x='step', y='entropy', hue='condition', style='head')
        plt.title('Attention Entropy Evolution (Higher = More Uniform)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entropy_evolution.png'), dpi=300)
        plt.close()

        # Plot concentration evolution
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=combined_df, x='step', y='concentration', hue='condition', style='head')
        plt.title('Attention Concentration Evolution (Higher = More Peaked)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'concentration_evolution.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="analysis/attention")
    args = parser.parse_args()

    compare_conditions(args.results_dir, args.output_dir)
