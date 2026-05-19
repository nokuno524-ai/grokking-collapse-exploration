import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any, List

def get_fourier_basis(prime: int) -> torch.Tensor:
    """
    Constructs a 1D discrete Fourier transform basis matrix of size (prime, prime).
    Returns a complex tensor.
    """
    n = torch.arange(prime)
    k = torch.arange(prime)
    nk = n.unsqueeze(1) * k.unsqueeze(0)
    basis = torch.exp(-2j * torch.pi * nk / prime)
    return basis

def compute_weight_fft(weights: torch.Tensor, prime: int) -> torch.Tensor:
    """
    Computes the Fast Fourier Transform of model weights across the vocab dimension.
    Assumes weights has shape (vocab_size, ...) where vocab_size >= prime.
    We only take the first `prime` elements.
    Returns the magnitude spectrum.
    """
    # Slice to prime to ignore special tokens if any, shape: (prime, ...)
    w_sliced = weights[:prime]

    # Compute FFT along the first dimension (vocab)
    w_fft = torch.fft.fft(w_sliced, dim=0)

    # Return magnitude
    return torch.abs(w_fft)

def analyze_attention_frequencies(
    attention_weights: torch.Tensor,
    prime: int
) -> torch.Tensor:
    """
    Analyzes the frequency spectrum of attention weights.
    attention_weights: (..., seq_len, seq_len)
    """
    # Assuming seq_len >= prime for relevant tasks, or we analyze patterns over a grid
    # For modular arithmetic (a + b), we typically embed inputs and compute attention.
    # This function is a generalized utility to run FFT over the last dimension.
    if attention_weights.size(-1) < prime:
        # Pad or just compute FFT on what's available
        fft_result = torch.fft.fft(attention_weights, dim=-1)
    else:
        fft_result = torch.fft.fft(attention_weights[..., :prime], dim=-1)

    return torch.abs(fft_result)

def compute_fourier_concentration(
    spectrum_magnitude: torch.Tensor,
    top_k: int = 5
) -> float:
    """
    Computes the concentration of energy in the top-k non-DC frequencies.
    spectrum_magnitude: (prime, ...)
    """
    # Average across all dimensions except the first (frequency)
    if spectrum_magnitude.dim() > 1:
        # Flatten all other dimensions and mean
        avg_spectrum = spectrum_magnitude.view(spectrum_magnitude.size(0), -1).mean(dim=1)
    else:
        avg_spectrum = spectrum_magnitude

    # Exclude DC component (index 0)
    avg_spectrum = avg_spectrum[1:]
    total_energy = avg_spectrum.sum().item()

    if total_energy < 1e-10:
        return 0.0

    top_energy = avg_spectrum.topk(min(top_k, len(avg_spectrum))).values.sum().item()
    return top_energy / total_energy

def plot_fourier_spectrum(
    spectrum_magnitude: torch.Tensor,
    output_path: str,
    title: str = "Fourier Spectrum of Weights",
    prime: Optional[int] = None
):
    """
    Plots the average magnitude of frequencies.
    """
    if spectrum_magnitude.dim() > 1:
        avg_spectrum = spectrum_magnitude.view(spectrum_magnitude.size(0), -1).mean(dim=1)
    else:
        avg_spectrum = spectrum_magnitude

    avg_spectrum = avg_spectrum.cpu().numpy()

    plt.figure(figsize=(10, 6))
    freqs = np.arange(len(avg_spectrum))

    # We typically plot up to Nyquist frequency (prime // 2) because spectrum is symmetric
    nyquist = len(avg_spectrum) // 2

    plt.bar(freqs[1:nyquist+1], avg_spectrum[1:nyquist+1], alpha=0.8, color='royalblue')
    plt.xlabel('Frequency (k)')
    plt.ylabel('Average Magnitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_frequency_evolution(
    spectra_history: List[torch.Tensor],
    steps: List[int],
    output_path: str,
    title: str = "Frequency Evolution over Training"
):
    """
    Plots a heatmap of frequency magnitudes over training steps.
    spectra_history: List of (prime,) or (prime, d) tensors.
    """
    # Aggregate to (num_steps, prime)
    agg_history = []
    for spec in spectra_history:
        if spec.dim() > 1:
            agg = spec.view(spec.size(0), -1).mean(dim=1)
        else:
            agg = spec
        agg_history.append(agg.cpu().numpy())

    # Stack into a 2D array (prime, num_steps)
    heatmap_data = np.stack(agg_history, axis=1)

    # Exclude DC and take up to Nyquist
    prime = heatmap_data.shape[0]
    nyquist = prime // 2
    heatmap_data = heatmap_data[1:nyquist+1, :]

    plt.figure(figsize=(12, 8))
    # We use imshow. origin='lower' puts freq k=1 at the bottom.
    plt.imshow(
        heatmap_data,
        aspect='auto',
        origin='lower',
        extent=[steps[0], steps[-1], 1, nyquist],
        cmap='viridis'
    )
    plt.colorbar(label='Magnitude')
    plt.xlabel('Training Steps')
    plt.ylabel('Frequency (k)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Analyze Fourier spectrum of a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--output-dir", type=str, default="analysis/fourier", help="Where to save plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config_dict = ckpt.get("config", {})

    try:
        from src.model import ModularArithmeticTransformer
    except ImportError:
        print("Could not import model. Run from repo root.")
        exit(1)

    model = ModularArithmeticTransformer(
        prime=config_dict.get("prime", 59),
        d_model=config_dict.get("d_model", 128),
        n_heads=config_dict.get("n_heads", 4),
        d_ff=config_dict.get("d_ff", 512),
        n_layers=config_dict.get("n_layers", 1),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print("Model loaded successfully.")

    # Get embedding weights
    embed_weight = model.token_embed.weight.detach()
    prime = config_dict.get("prime", 59)

    spectrum = compute_weight_fft(embed_weight, prime)
    concentration = compute_fourier_concentration(spectrum)
    print(f"Fourier Concentration (top-5): {concentration:.4f}")

    plot_fourier_spectrum(
        spectrum,
        os.path.join(args.output_dir, "fourier_spectrum.png"),
        title=f"Embedding Fourier Spectrum (Concentration: {concentration:.3f})"
    )
    print(f"Saved plot to {args.output_dir}")
