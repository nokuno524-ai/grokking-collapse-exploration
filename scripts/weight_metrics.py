import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import math

import torch


def compute_frobenius_norm(weight: torch.Tensor) -> float:
    """Compute the Frobenius norm of a weight matrix."""
    return torch.norm(weight, p='fro').item()


def compute_spectral_norm(weight: torch.Tensor) -> float:
    """Compute the spectral norm (L2 matrix norm) of a weight matrix."""
    if weight.ndim < 2:
        # For 1D tensors like biases or layer norm weights, spectral norm is just the vector norm
        return torch.norm(weight, p=2).item()
    return torch.linalg.matrix_norm(weight, ord=2).item()


def compute_effective_rank(weight: torch.Tensor) -> float:
    """
    Compute the effective rank of a weight matrix.
    Effective rank is defined as exp(Shannon entropy of normalized singular values).
    """
    if weight.ndim < 2:
        return 1.0 # Effective rank of a vector is 1

    # For multi-dimensional tensors, reshape to 2D
    if weight.ndim > 2:
        weight = weight.reshape(weight.shape[0], -1)

    s = torch.linalg.svdvals(weight)
    s_sum = s.sum()
    if s_sum < 1e-10:
        return 0.0

    s_norm = s / s_sum
    # Calculate Shannon entropy
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    return math.exp(entropy.item())


def compute_activation_condition(activations: torch.Tensor) -> float:
    """
    Compute condition metric (condition number) of the activation covariance matrix.
    Args:
        activations: Tensor of shape (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim).
    """
    if activations.ndim > 2:
        activations = activations.reshape(-1, activations.shape[-1])

    # Center the activations
    centered = activations - activations.mean(dim=0, keepdim=True)
    # Compute covariance matrix
    cov = (centered.T @ centered) / (centered.shape[0] - 1 + 1e-8)

    # Condition number = max_singular_value / min_singular_value
    try:
        s = torch.linalg.svdvals(cov)
        if s.shape[0] == 0:
            return 1.0
        min_s = s[-1].item()
        if min_s < 1e-10:
            return float('inf')
        return (s[0] / min_s).item()
    except RuntimeError:
        return float('nan')


def process_checkpoint(checkpoint_path: Path, activations_dict: Dict[str, torch.Tensor] = None) -> List[Dict[str, Any]]:
    """
    Process a single checkpoint and compute weight metrics for each layer.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt).
        activations_dict: Optional dictionary of cached activations for condition metrics.

    Returns:
        List of dictionaries, each containing metrics for a specific layer.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    step = ckpt.get('step', -1)

    # Try to extract condition from path if possible (e.g. results/pure/checkpoint_100.pt)
    condition = checkpoint_path.parent.name

    metrics = []

    model_state = ckpt.get('model_state', {})
    if not model_state:
        # Fallback if the checkpoint is just a state dict
        model_state = ckpt

    for layer_name, weight in model_state.items():
        if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
            continue

        fro_norm = compute_frobenius_norm(weight)
        spec_norm = compute_spectral_norm(weight)
        eff_rank = compute_effective_rank(weight)

        metrics.append({
            'checkpoint_path': str(checkpoint_path),
            'condition': condition,
            'step': step,
            'layer': layer_name,
            'metric_name': 'frobenius_norm',
            'metric_value': fro_norm
        })
        metrics.append({
            'checkpoint_path': str(checkpoint_path),
            'condition': condition,
            'step': step,
            'layer': layer_name,
            'metric_name': 'spectral_norm',
            'metric_value': spec_norm
        })
        metrics.append({
            'checkpoint_path': str(checkpoint_path),
            'condition': condition,
            'step': step,
            'layer': layer_name,
            'metric_name': 'effective_rank',
            'metric_value': eff_rank
        })

        # If activations are provided for this layer, compute the activation condition metric
        if activations_dict is not None and layer_name in activations_dict:
            act = activations_dict[layer_name]
            act_cond = compute_activation_condition(act)
            metrics.append({
                'checkpoint_path': str(checkpoint_path),
                'condition': condition,
                'step': step,
                'layer': layer_name,
                'metric_name': 'activation_condition',
                'metric_value': act_cond
            })

    return metrics


def analyze_checkpoint_dir(checkpoint_dir: Path, output_csv: Path) -> None:
    """
    Analyze all checkpoints in a directory and write metrics to a CSV.

    Args:
        checkpoint_dir: Directory containing .pt checkpoint files.
        output_csv: Path to the output CSV file.
    """
    all_metrics = []

    checkpoint_files = list(checkpoint_dir.glob('checkpoint_*.pt'))
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Sort by step number
    def extract_step(path: Path) -> int:
        try:
            return int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            return -1

    checkpoint_files.sort(key=extract_step)

    for ckpt_path in checkpoint_files:
        try:
            metrics = process_checkpoint(ckpt_path)
            all_metrics.extend(metrics)
        except Exception as e:
            print(f"Failed to process {ckpt_path}: {e}")

    if not all_metrics:
        print("No metrics computed.")
        return

    fieldnames = ['checkpoint_path', 'condition', 'step', 'layer', 'metric_name', 'metric_value']

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    print(f"Wrote metrics for {len(checkpoint_files)} checkpoints to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compute weight metrics from checkpoints.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory containing checkpoint .pt files")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to output CSV file")

    args = parser.parse_args()

    analyze_checkpoint_dir(Path(args.checkpoint_dir), Path(args.output_csv))


if __name__ == "__main__":
    main()
