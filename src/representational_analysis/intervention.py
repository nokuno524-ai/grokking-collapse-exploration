import torch
import torch.nn as nn
from typing import Dict, Optional, Callable

class ActivationSteeringHook:
    """
    Injects task-relevant representational structure at specific layers
    by perturbing activations during the forward pass.
    """
    def __init__(self, steering_vector: torch.Tensor, scale: float = 1.0):
        self.steering_vector = steering_vector
        self.scale = scale
        self.handle = None

    def __call__(self, module, input, output):
        # output shape is typically (batch_size, seq_len, d_model)
        # We steer the representations by adding a scaled vector
        if isinstance(output, tuple):
            steered = output[0] + self.scale * self.steering_vector.to(output[0].device)
            return (steered,) + output[1:]
        else:
            return output + self.scale * self.steering_vector.to(output.device)

    def register(self, module: nn.Module):
        self.handle = module.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

class GeometricRegularizer:
    """
    Regularizes weights to maintain specific geometric properties
    (e.g., Fourier concentration, effective rank) during training.
    """
    def __init__(self, target_rank: float = None, rank_weight: float = 0.01,
                 fourier_weight: float = 0.01):
        self.target_rank = target_rank
        self.rank_weight = rank_weight
        self.fourier_weight = fourier_weight

    def compute_embedding_rank(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        s = torch.linalg.svdvals(weight_matrix)
        s = s / (s.sum() + 1e-10)
        entropy = -(s * torch.log(s + 1e-10)).sum()
        return torch.exp(entropy)

    def compute_fourier_concentration(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        # Assumes weight_matrix shape (vocab_size, dim)
        spectrum = torch.fft.fft(weight_matrix, dim=0).abs() ** 2
        # Maximize concentration (energy in specific frequencies)
        # Simply penalize uniform energy distribution
        total_energy = spectrum.sum(dim=0, keepdim=True) + 1e-10
        norm_spectrum = spectrum / total_energy
        entropy = -(norm_spectrum * torch.log(norm_spectrum + 1e-10)).sum()
        return entropy.mean()

    def __call__(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)

        if hasattr(model, 'token_embed'):
            W = model.token_embed.weight

            # 1. Rank regularization
            if self.target_rank is not None:
                current_rank = self.compute_embedding_rank(W)
                loss += self.rank_weight * (current_rank - self.target_rank)**2

            # 2. Fourier concentration regularization (encourage sparsity in frequency domain)
            if self.fourier_weight > 0:
                fourier_entropy = self.compute_fourier_concentration(W)
                loss += self.fourier_weight * fourier_entropy

        return loss

def initialize_with_geometry(model: nn.Module, seed_matrix: Optional[torch.Tensor] = None,
                             scale: float = 1.0, add_noise: float = 0.01):
    """
    Pre-initialize the model's embeddings with a grokking-favorable geometry
    (e.g., strong Fourier components).
    """
    if not hasattr(model, 'token_embed'):
        raise ValueError("Model does not have token_embed attribute")

    vocab_size, d_model = model.token_embed.weight.shape
    device = model.token_embed.weight.device

    if seed_matrix is not None:
        assert seed_matrix.shape == (vocab_size, d_model)
        init_weight = seed_matrix.clone().to(device)
    else:
        # Create a circle/Fourier basis geometry
        init_weight = torch.zeros((vocab_size, d_model), device=device)

        # Use first few dimensions for primary frequencies
        for k in range(1, min(d_model//2, 5)):
            freq = 2 * torch.pi * k / vocab_size
            positions = torch.arange(vocab_size, device=device, dtype=torch.float)

            idx1 = 2*k - 2
            idx2 = 2*k - 1
            if idx2 < d_model:
                init_weight[:, idx1] = torch.cos(freq * positions)
                init_weight[:, idx2] = torch.sin(freq * positions)

    init_weight = init_weight * scale

    if add_noise > 0:
        init_weight += torch.randn_like(init_weight) * add_noise

    with torch.no_grad():
        model.token_embed.weight.copy_(init_weight)
