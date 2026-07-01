import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path
import json

# Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets, Power et al. 2022.

class WeightAnalysisSuite:
    """
    Suite of tools to analyze weight matrices during training, helping to understand
    how model collapse affects grokking (disrupting weight norm growth).
    Inspired by 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets' (Power et al. 2022).
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def get_weight_norm(self) -> float:
        """
        Track L2 norm of all weight matrices per layer across training.
        """
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_norm += param.norm(2).item() ** 2
        return total_norm ** 0.5

    def get_effective_rank(self, weight_matrix: torch.Tensor) -> float:
        """
        Compute effective rank via SVD (entropy of normalized singular values).
        Tracks rank collapse/expansion.
        """
        if weight_matrix.dim() < 2:
            return 0.0

        with torch.no_grad():
            if weight_matrix.dim() > 2:
                # Flatten spatial dimensions for convs, or just reshape to 2D
                weight_matrix = weight_matrix.view(weight_matrix.size(0), -1)

            s = torch.linalg.svdvals(weight_matrix)
            s = s / s.sum()
            entropy = -(s * torch.log(s + 1e-10)).sum()
            return torch.exp(entropy).item()

    def get_all_effective_ranks(self) -> Dict[str, float]:
        """
        Get effective ranks for all 2D weight matrices in the model.
        """
        ranks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                ranks[name] = self.get_effective_rank(param.detach())
        return ranks

    def get_weight_connectivity(self, layer1_name: str, layer2_name: str) -> float:
        """
        Weight connectivity: cosine similarity between layers, detecting layer convergence/divergence.
        Assumes layers can be flattened to same size or we compare their outputs.
        Here we compare flattened parameter vectors if sizes match.
        """
        params = dict(self.model.named_parameters())
        if layer1_name not in params or layer2_name not in params:
            raise ValueError("Layer names not found in model.")

        w1 = params[layer1_name].detach().flatten()
        w2 = params[layer2_name].detach().flatten()

        if w1.shape != w2.shape:
            # If shapes don't match, this specific connectivity metric is undefined for direct params
            return 0.0

        cos_sim = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))
        return cos_sim.item()

    def estimate_hessian_eigenvalues(
        self,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        k: int = 1,
        num_iters: int = 10
    ) -> List[float]:
        """
        Hessian eigenvalue estimation: compute top-K eigenvalues of loss Hessian
        (via power iteration) to detect sharp/flat minima.
        """
        # simplified power iteration for top-1 eigenvalue
        # using Hutchinson's method / Hessian-vector products

        self.model.eval()

        # Initial random vector for power iteration
        v = [torch.randn_like(p) for p in self.model.parameters() if p.requires_grad]

        # Normalize v
        norm = torch.sqrt(sum(torch.sum(x**2) for x in v))
        v = [x / norm for x in v]

        eigenvalues = []

        for _ in range(k):
            eigenval = 0.0
            for _ in range(num_iters):
                # 1. Forward pass
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

                # 2. First derivative
                grads = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], create_graph=True, retain_graph=True)

                # 3. Hessian-vector product
                # grad^T * v
                grad_v = sum(torch.sum(g * x) for g, x in zip(grads, v))

                # Hv = d(grad^T * v)/dw
                Hv = torch.autograd.grad(grad_v, [p for p in self.model.parameters() if p.requires_grad], retain_graph=True)

                # Rayleigh quotient (eigenvalue estimate)
                eigenval = sum(torch.sum(h * x) for h, x in zip(Hv, v)).item()

                # Update v for next iteration
                norm = torch.sqrt(sum(torch.sum(h**2) for h in Hv))
                if norm.item() == 0:
                    break
                v = [h / norm for h in Hv]

            eigenvalues.append(eigenval)
            # Deflation for subsequent eigenvalues omitted for brevity in this simple top-1 loop,
            # but would project out the found eigenvector.

        return eigenvalues

    def analyze_gradient_flow(self) -> Dict[str, float]:
        """
        Gradient flow analysis: gradient norm per layer, vanishing/exploding gradient detection.
        Assumes backward pass has just been called.
        """
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms[name] = param.grad.norm(2).item()
        return grad_norms

    @classmethod
    def load_from_checkpoint_dir(cls, model: nn.Module, checkpoint_dir: str, step: int) -> 'WeightAnalysisSuite':
        """
        Support loading from checkpoint directories.
        In this repository, checkpoint files save weights using the key 'model_state'.
        """
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        return cls(model)
