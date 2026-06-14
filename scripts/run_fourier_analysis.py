"""
Script to run Fourier, composition, and gradient analysis on experiment checkpoints.
"""

import os
import glob
import json
import torch
import numpy as np
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModularArithmeticTransformer
from src.data import get_dataloaders
from src.fourier_analysis import (
    extract_fourier_basis,
    identify_dominant_frequencies,
    plot_fourier_spectrum
)
from src.composition import composition_matrix, detect_circuits
from src.gradient_analysis import compute_gradient_noise_scale

def find_checkpoints(result_dir):
    checkpoints = glob.glob(os.path.join(result_dir, "checkpoint_*.pt"))
    # Sort by step number
    def extract_step(path):
        filename = os.path.basename(path)
        step_str = filename.split('_')[1].split('.')[0]
        return int(step_str)

    checkpoints.sort(key=extract_step)
    return checkpoints

def load_model_from_checkpoint(checkpoint_path, prime=59):
    model = ModularArithmeticTransformer(prime=prime)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def run_analysis_for_dir(result_dir, prime=59):
    print(f"Analyzing {result_dir}...")
    checkpoints = find_checkpoints(result_dir)
    if not checkpoints:
        print(f"No checkpoints found in {result_dir}")
        return

    # Use the final checkpoint for most intensive analysis
    final_ckpt_path = checkpoints[-1]
    model = load_model_from_checkpoint(final_ckpt_path, prime=prime)

    # 1. Fourier Analysis
    spectrum = extract_fourier_basis(model)
    dominant_freqs = identify_dominant_frequencies(model, threshold=0.5)

    plot_path = os.path.join(result_dir, "fourier_spectrum.png")
    plot_fourier_spectrum(spectrum, title="Embedding Fourier Spectrum", save_path=plot_path)

    # 2. Composition Analysis
    # Get a dataloader for this (using dummy clean data for analysis)
    train_dl, test_dl = get_dataloaders(prime=prime, train_frac=0.3, batch_size=512)

    comp_matrix = composition_matrix(model, test_dl)
    circuits = detect_circuits(comp_matrix, threshold=0.1)

    # 3. Gradient Analysis
    # We only do this if we want to run backward, which requires train mode
    noise_scale = compute_gradient_noise_scale(model, test_dl)

    # 4. Save summaries
    summary = {
        "final_checkpoint": final_ckpt_path,
        "dominant_frequencies": dominant_freqs,
        "circuit_edges": [(int(u), int(v)) for u, v in circuits],
        "gradient_noise_scale": float(noise_scale)
    }

    with open(os.path.join(result_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Finished analysis for {result_dir}. Saved to analysis_summary.json and fourier_spectrum.png")

if __name__ == "__main__":
    # If a specific directory is provided, use it, otherwise analyze results/pure (if it exists)
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        run_analysis_for_dir(target_dir)
    else:
        # Search for results/pure
        base_dir = "results"
        if os.path.exists(base_dir):
            for condition in os.listdir(base_dir):
                cond_dir = os.path.join(base_dir, condition)
                if os.path.isdir(cond_dir):
                    # check for seeds
                    for seed in os.listdir(cond_dir):
                        seed_dir = os.path.join(cond_dir, seed)
                        if os.path.isdir(seed_dir):
                            run_analysis_for_dir(seed_dir)
