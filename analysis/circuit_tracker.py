import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Progress Measures for Grokking via Mechanistic Interpretability, Nanda et al. 2023.
# Quantifying LLM Attention-Head Stability (middle layers are least stable).

class CircuitTracker:
    """
    Tracks the formation of specific circuits across training checkpoints and identifies
    'circuit emergence' steps where attention head connectivity stabilizes.
    Inspired by Nanda et al. 2023 on progress measures for grokking, and
    recent work on 'Quantifying LLM Attention-Head Stability' (showing middle layers are least stable).
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.head_importance_history: Dict[int, Dict[int, float]] = {}  # step -> head_idx -> importance
        self.circuit_stability_history: Dict[int, float] = {}

    def compute_stability_score(self, current_attn_patterns: torch.Tensor, prev_attn_patterns: Optional[torch.Tensor]) -> float:
        """
        Compute circuit stability score. Inspired by 'Quantifying LLM Attention-Head Stability'.
        Middle layers are often least stable, but we compute stability as the cosine similarity
        or negative change between consecutive checkpoints.

        Args:
            current_attn_patterns: Tensor of shape (batch, n_heads, seq_len, seq_len)
            prev_attn_patterns: Tensor of same shape from the previous step.
        """
        if prev_attn_patterns is None:
            return 0.0

        # Flatten and compute cosine similarity
        curr_flat = current_attn_patterns.flatten(start_dim=1)
        prev_flat = prev_attn_patterns.flatten(start_dim=1)

        # Mean cosine similarity across batch
        cos_sim = torch.nn.functional.cosine_similarity(curr_flat, prev_flat, dim=1).mean().item()
        return cos_sim

    def measure_head_importance(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        ablation_type: str = "zero"
    ) -> Dict[int, float]:
        """
        Measure head importance via activation patching: zero ablation or mean ablation.
        We check the effect on output logits (specifically, loss difference).

        Args:
            inputs: Tensor of inputs
            targets: Tensor of targets
            ablation_type: 'zero' or 'mean'

        Returns:
            Dictionary mapping head index to importance score (loss increase).
        """
        original_loss = self._get_loss(inputs, targets)
        n_heads = getattr(self.model, "n_heads", 1)
        d_model = getattr(self.model, "d_model", 128)
        head_dim = d_model // n_heads

        importance_scores = {}

        for head_idx in range(n_heads):
            # For each head, we will zero out the out_proj weights in the multi-head attention layer
            # for the columns that correspond to this head's output.
            # In our ModularArithmeticTransformer, the attention layer is inside the transformer encoder layer.
            # Assuming n_layers=1, we access self.model.transformer.layers[0].self_attn.out_proj.weight
            try:
                layer = self.model.transformer.layers[0]
                out_proj = layer.self_attn.out_proj
                original_weight = out_proj.weight.data.clone()

                # The output projection weight has shape (d_model, d_model)
                # The input to this projection comes from concatenating head outputs.
                # So head_idx output corresponds to columns: head_idx * head_dim to (head_idx + 1) * head_dim
                start_col = head_idx * head_dim
                end_col = (head_idx + 1) * head_dim

                if ablation_type == "zero":
                    out_proj.weight.data[:, start_col:end_col] = 0.0
                elif ablation_type == "mean":
                    mean_val = original_weight[:, start_col:end_col].mean()
                    out_proj.weight.data[:, start_col:end_col] = mean_val

                ablated_loss = self._get_loss(inputs, targets)
                importance_scores[head_idx] = float(ablated_loss - original_loss)

                # Restore original weights
                out_proj.weight.data.copy_(original_weight)
            except AttributeError:
                importance_scores[head_idx] = 0.0

        return importance_scores

    def _get_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss.item()

    def track_checkpoint(self, step: int, inputs: torch.Tensor, targets: torch.Tensor, prev_attn_patterns: Optional[torch.Tensor] = None) -> Tuple[float, Dict[int, float], torch.Tensor]:
        """
        Process a single checkpoint, extracting stability and importance.
        """
        # Mocking attention patterns for stability computation
        # In practice, extract via model hooks
        batch_size = inputs.size(0)
        n_heads = getattr(self.model, "n_heads", 4)
        seq_len = inputs.size(1) if inputs.dim() > 1 else 1

        current_attn = torch.rand(batch_size, n_heads, seq_len, seq_len, device=self.device)

        stability = self.compute_stability_score(current_attn, prev_attn_patterns)
        self.circuit_stability_history[step] = stability

        importance = self.measure_head_importance(inputs, targets, ablation_type="zero")
        self.head_importance_history[step] = importance

        return stability, importance, current_attn

    def identify_emergence_steps(self, threshold: float = 0.9) -> List[int]:
        """
        Identify 'circuit emergence' steps where attention head connectivity stabilizes.
        """
        emergence_steps = []
        for step, stability in sorted(self.circuit_stability_history.items()):
            if stability >= threshold:
                emergence_steps.append(step)
        return emergence_steps

    def plot_evolution(self, output_path: str):
        """
        Output per-head importance evolution charts.
        Track which heads are 'recruited' during grokking vs which remain dormant.
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping plot.")
            return

        steps = sorted(self.head_importance_history.keys())
        if not steps:
            return

        n_heads = len(self.head_importance_history[steps[0]])

        plt.figure(figsize=(10, 6))
        for head_idx in range(n_heads):
            importances = [self.head_importance_history[s].get(head_idx, 0) for s in steps]
            plt.plot(steps, importances, label=f"Head {head_idx}")

        plt.title("Per-Head Importance Evolution (Circuit Formation)")
        plt.xlabel("Training Step")
        plt.ylabel("Importance (Ablation Loss Increase)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved evolution plot to {output_path}")
