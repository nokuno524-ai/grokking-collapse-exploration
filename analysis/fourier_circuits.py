import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Tuple
from pathlib import Path

def get_attention_grid(model, prime: int, device: str = 'cpu') -> torch.Tensor:
    """
    Computes attention weights for all possible (a, b) inputs.
    Returns:
        attn_grid: shape (n_heads, seq_len, seq_len, prime, prime)
    """
    model.eval()
    model.to(device)

    # Generate all pairs
    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    x = torch.tensor(all_pairs, device=device) # (p*p, 2)

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=device).unsqueeze(0).expand(x.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        layer = model.transformer.layers[0]
        # need_weights=True returns (batch, n_heads, seq_len, seq_len) if average_attn_weights=False
        _, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)

        # attn_weights is (p*p, n_heads, 2, 2)
        attn_grid = attn_weights.view(prime, prime, model.n_heads, 2, 2)
        # Permute to (n_heads, seq_len, seq_len, prime, prime)
        attn_grid = attn_grid.permute(2, 3, 4, 0, 1)

    return attn_grid

def get_2d_fourier_transform(grid: torch.Tensor) -> torch.Tensor:
    """
    Computes 2D Fourier transform of the attention grid over the input grid (a,b).
    Args:
        grid: shape (..., prime, prime)
    Returns:
        fft_mag: magnitude of FFT, shape (..., prime, prime)
    """
    # Compute 2D FFT over the last two dimensions
    fft = torch.fft.fft2(grid, dim=(-2, -1))
    return torch.abs(fft)

def analyze_fourier_circuits(checkpoint_path: str, model, prime: int) -> Dict:
    """
    Analyzes Fourier components for a given checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    attn_grid = get_attention_grid(model, prime, device)
    fft_mag = get_2d_fourier_transform(attn_grid)

    # Dominant frequencies: sort the magnitudes
    # To identify key frequencies, we can just return the top k or the full spectrum
    return {
        'step': ckpt['step'],
        'fft_mag': fft_mag.cpu(), # (n_heads, 2, 2, p, p)
    }

def plot_fourier_heatmaps(history: List[Dict], output_dir: str):
    """
    Plots heatmap of Fourier coefficients over training.
    """
    os.makedirs(output_dir, exist_ok=True)

    steps = [h['step'] for h in history]

    if len(history) == 0:
        return

    n_heads = history[0]['fft_mag'].shape[0]
    prime = history[0]['fft_mag'].shape[-1]

    for head_idx in range(n_heads):
        for q_pos in range(2):
            for k_pos in range(2):
                # Shape: (num_steps, prime, prime)
                mag_history = torch.stack([h['fft_mag'][head_idx, q_pos, k_pos] for h in history])
                mag_history[:, 0, 0] = 0 # zero out DC component

                # Instead of a line plot, let's create a heatmap over time for the top frequencies
                # Average the spectrum over all frequencies along one dimension to get a 1D spectrum per step
                # Or simply take the maximum frequency in the grid to flatten to 1D
                spec_1d = mag_history.amax(dim=1) # Shape: (num_steps, prime)

                plt.figure(figsize=(10, 6))
                # x-axis: frequencies, y-axis: steps
                im = plt.imshow(spec_1d.numpy(), aspect='auto', cmap='viridis', origin='lower',
                                extent=[0, prime, steps[0], steps[-1]])
                plt.title(f'Head {head_idx} Attn({q_pos}->{k_pos}) Fourier Magnitude over Time')
                plt.xlabel('Frequency Component')
                plt.ylabel('Step')
                plt.colorbar(im)
                plt.savefig(os.path.join(output_dir, f'fourier_heatmap_head_{head_idx}_{q_pos}_{k_pos}.png'))
                plt.close()

def compare_runs(run1_dir: str, run2_dir: str, model_cls, prime: int, output_dir: str):
    """
    Compares grokked vs collapsed runs.
    """
    # Assuming run1 is grokked, run2 is collapsed
    model1 = model_cls(prime=prime)
    model2 = model_cls(prime=prime)

    # Load last checkpoint
    ckpts1 = sorted(glob.glob(os.path.join(run1_dir, "checkpoint_*.pt")), key=os.path.getmtime)
    ckpts2 = sorted(glob.glob(os.path.join(run2_dir, "checkpoint_*.pt")), key=os.path.getmtime)

    if not ckpts1 or not ckpts2:
        print("Missing checkpoints for comparison.")
        return

    res1 = analyze_fourier_circuits(ckpts1[-1], model1, prime)
    res2 = analyze_fourier_circuits(ckpts2[-1], model2, prime)

    # Compare
    # Let's plot the spectrum of the first head (0,0) for both
    # Average across all heads and positions
    spec1 = res1['fft_mag'].mean(dim=(0,1,2))
    spec2 = res2['fft_mag'].mean(dim=(0,1,2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(spec1.numpy(), cmap='viridis')
    ax1.set_title('Grokked Model - Avg Fourier Spectrum')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(spec2.numpy(), cmap='viridis')
    ax2.set_title('Collapsed Model - Avg Fourier Spectrum')
    plt.colorbar(im2, ax=ax2)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fourier_comparison.png'))
    plt.close()
