import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import defaultdict
import os

class GradientTracker:
    """
    Tracks and analyzes per-layer gradients during training to identify
    vanishing/exploding gradients and compute gradient SNR.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.grad_history = defaultdict(list)
        self.grad_variance_history = defaultdict(list)
        self.snr_history = defaultdict(list)
        self.steps = []

    def step(self, step_idx: int):
        """
        Record gradient statistics at the current step.
        Call this immediately after loss.backward() and before optimizer.step().
        """
        self.steps.append(step_idx)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # Compute norm
                norm = grad.norm(2).item()
                self.grad_history[name].append(norm)

                # Compute variance and SNR
                mean = grad.mean().item()
                var = grad.var().item() if grad.numel() > 1 else 0.0
                self.grad_variance_history[name].append(var)

                # SNR = mean^2 / variance
                snr = (mean ** 2) / var if var > 1e-10 else 0.0
                self.snr_history[name].append(snr)

    def check_vanishing_exploding(self, step_idx: int, vanishing_thresh: float = 1e-5, exploding_thresh: float = 1e2) -> Dict[str, str]:
        """
        Identify layers with vanishing or exploding gradients at the current step.
        Returns a dictionary mapping layer name to 'vanishing' or 'exploding'.
        """
        issues = {}
        for name in self.grad_history:
            if not self.grad_history[name]:
                continue
            current_norm = self.grad_history[name][-1]
            if current_norm < vanishing_thresh:
                issues[name] = 'vanishing'
            elif current_norm > exploding_thresh:
                issues[name] = 'exploding'
        return issues

    def save_results(self, output_dir: str):
        """Save tracked metrics as numpy arrays."""
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "grad_steps.npy"), np.array(self.steps))

        # Convert to arrays
        grad_norms = {k: np.array(v) for k, v in self.grad_history.items()}
        grad_vars = {k: np.array(v) for k, v in self.grad_variance_history.items()}
        grad_snr = {k: np.array(v) for k, v in self.snr_history.items()}

        np.save(os.path.join(output_dir, "grad_norms.npy"), grad_norms)
        np.save(os.path.join(output_dir, "grad_vars.npy"), grad_vars)
        np.save(os.path.join(output_dir, "grad_snr.npy"), grad_snr)

    def plot_gradient_norms(self, output_path: str):
        """Generate a plot of gradient norms over time."""
        plt.figure(figsize=(12, 8))
        for name, history in self.grad_history.items():
            plt.plot(self.steps, history, label=name, alpha=0.7)

        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm (L2)')
        plt.title('Gradient Norms Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_snr(self, output_path: str):
        """Generate a plot of gradient Signal-to-Noise Ratio (SNR) over time."""
        plt.figure(figsize=(12, 8))
        for name, history in self.snr_history.items():
            plt.plot(self.steps, history, label=name, alpha=0.7)

        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient SNR (Mean^2 / Variance)')
        plt.title('Gradient Signal-to-Noise Ratio (SNR) Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
