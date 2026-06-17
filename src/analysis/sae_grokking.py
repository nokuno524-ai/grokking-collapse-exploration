"""
Sparse Autoencoder (SAE) analysis for tracking feature emergence during grokking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SparseAutoencoder(nn.Module):
    """
    1-layer Sparse Autoencoder for extracting interpretable features from model activations.
    """
    def __init__(self, d_in: int, d_hidden: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        # Tie encoder and decoder weights optionally, but standard SAE doesn't require it
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Normalize decoder weights
        nn.init.orthogonal_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Centering
        x_centered = x - self.b_dec

        # Encoding
        f = F.relu(self.encoder(x_centered))

        # Decoding
        x_reconstructed = self.decoder(f) + self.b_dec

        # Loss computation
        l2_loss = F.mse_loss(x_reconstructed, x)
        l1_loss = f.abs().sum(-1).mean()
        loss = l2_loss + self.l1_coeff * l1_loss

        return x_reconstructed, f, loss


def extract_activations(model: nn.Module, dataloader, device: torch.device) -> torch.Tensor:
    """
    Extract internal representations (e.g., output of the transformer layer) from a model.
    """
    model.eval()
    activations = []

    # Hook to capture transformer output
    def hook_fn(module, input, output):
        # Handle tuple output from TransformerEncoderLayer if any
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        # Average over sequence length (positions)
        # Check if layer uses batch_first
        if getattr(module, 'batch_first', False):
            activations.append(act.mean(dim=1).detach().cpu())
        else:
            # If batch_first=False, act is (seq_len, batch_size, dim)
            activations.append(act.mean(dim=0).detach().cpu())

    # Register hook on the transformer layer
    layer = model.transformer.layers[0] if hasattr(model.transformer, 'layers') else list(model.transformer.children())[0]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, _ in dataloader:
            model(inputs.to(device))

    handle.remove()
    return torch.cat(activations, dim=0)


def train_sae(
    activations: torch.Tensor,
    d_hidden: int,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    l1_coeff: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on a dataset of activations.
    """
    d_in = activations.shape[-1]
    sae = SparseAutoencoder(d_in, d_hidden, l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            _, _, loss = sae(x)
            loss.backward()

            optimizer.step()

            # Normalize decoder weights AFTER optimizer step
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, p=2, dim=0)

    return sae


def track_sae_features(
    models_dict: Dict[str, nn.Module],
    dataloader,
    device: torch.device,
    d_hidden: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Track feature emergence across different phases/models.
    models_dict: Dict of phase -> model. e.g. {'pre': model1, 'grok': model2, 'post': model3}
    Returns average feature activation frequencies.
    """
    results = {}

    for phase, model in models_dict.items():
        acts = extract_activations(model, dataloader, device)
        sae = train_sae(acts, d_hidden, epochs=5, device=device)

        sae.eval()
        with torch.no_grad():
            _, f, _ = sae(acts.to(device))
            # Fraction of inputs where feature is active > 0
            feature_freq = (f > 1e-4).float().mean(dim=0).cpu()
            results[phase] = feature_freq

    return results

def correlate_with_circuit(feature_activations: torch.Tensor, circuit_scores: torch.Tensor) -> torch.Tensor:
    """
    Correlate SAE feature activations with circuit discovery scores (e.g. from circuit_discovery.py).
    If circuit_discovery.py is present and provides a node score, we compute Pearson correlation.
    For this implementation, we calculate correlation against a provided tensor of circuit scores.
    """
    # Standardize both
    f_norm = feature_activations - feature_activations.mean(dim=0, keepdim=True)
    f_norm = f_norm / (f_norm.std(dim=0, keepdim=True) + 1e-8)

    c_norm = circuit_scores - circuit_scores.mean()
    c_norm = c_norm / (c_norm.std() + 1e-8)

    # Pearson correlation
    correlation = (f_norm * c_norm.unsqueeze(1)).mean(dim=0)
    return correlation

def compute_feature_stability(feature_activations_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute feature stability across multiple seeds/models.
    feature_activations_list: List of feature activation tensors, one per seed.
    Returns: Variance or stability score per feature.
    """
    # Stack activations across seeds: (num_seeds, num_samples, num_features)
    # We assume features are aligned (e.g. via matching/Hungarian algorithm) or we just compute variance of frequency.
    # For a simple stability metric, we compute the variance of the feature activation frequencies across seeds.
    freqs = []
    for acts in feature_activations_list:
        freq = (acts > 1e-4).float().mean(dim=0)
        freqs.append(freq)

    freqs_tensor = torch.stack(freqs, dim=0)
    stability = 1.0 / (freqs_tensor.var(dim=0) + 1e-5) # Inverse variance
    return stability
