import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Any, List
import csv
from pathlib import Path
from src.model import ModularArithmeticTransformer

@contextmanager
def ablate_head(model: ModularArithmeticTransformer, layer_idx: int, head_idx: int):
    """
    Context manager that temporarily ablates (zeroes out) a specific attention head
    by setting its output projection weights to zero.

    Args:
        model: The transformer model.
        layer_idx: The index of the transformer layer.
        head_idx: The index of the head to ablate.
    """
    if not 0 <= layer_idx < len(model.transformer.layers):
        raise ValueError(f"Invalid layer index: {layer_idx}")
    if not 0 <= head_idx < model.n_heads:
        raise ValueError(f"Invalid head index: {head_idx}")

    layer = model.transformer.layers[layer_idx]
    out_proj = layer.self_attn.out_proj

    # Store original weights
    orig_weight = out_proj.weight.data.clone()

    head_dim = model.d_model // model.n_heads
    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    try:
        # Zero out the weights for this head
        # out_proj weight shape is (d_model, d_model)
        # The input to out_proj comes from concatenated head outputs
        out_proj.weight.data[:, start_idx:end_idx] = 0.0
        yield
    finally:
        # Restore original weights
        out_proj.weight.data.copy_(orig_weight)

@contextmanager
def ablate_mlp_neuron(model: ModularArithmeticTransformer, layer_idx: int, neuron_idx: int):
    """
    Context manager that temporarily ablates a specific MLP neuron
    by setting its incoming and outgoing weights to zero.

    Args:
        model: The transformer model.
        layer_idx: The index of the transformer layer.
        neuron_idx: The index of the MLP neuron to ablate.
    """
    if not 0 <= layer_idx < len(model.transformer.layers):
        raise ValueError(f"Invalid layer index: {layer_idx}")

    layer = model.transformer.layers[layer_idx]
    linear1 = layer.linear1
    linear2 = layer.linear2

    if not 0 <= neuron_idx < linear1.out_features:
        raise ValueError(f"Invalid neuron index: {neuron_idx}")

    # Store original weights and biases
    orig_l1_weight = linear1.weight.data.clone()
    orig_l1_bias = linear1.bias.data.clone() if linear1.bias is not None else None
    orig_l2_weight = linear2.weight.data.clone()

    try:
        linear1.weight.data[neuron_idx, :] = 0.0
        if linear1.bias is not None:
            linear1.bias.data[neuron_idx] = 0.0
        linear2.weight.data[:, neuron_idx] = 0.0
        yield
    finally:
        linear1.weight.data.copy_(orig_l1_weight)
        if linear1.bias is not None:
            linear1.bias.data.copy_(orig_l1_bias)
        linear2.weight.data.copy_(orig_l2_weight)

def score_head_importances(
    model: ModularArithmeticTransformer,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_csv: Path
) -> List[Dict[str, Any]]:
    """
    Ablate each head one by one and measure accuracy delta.
    Saves the results to a CSV file.

    Args:
        model: The model to analyze.
        test_loader: DataLoader for the test set.
        device: Torch device.
        output_csv: Path to save the output CSV.

    Returns:
        List of dictionaries with keys: 'layer', 'head', 'baseline_acc', 'ablated_acc', 'acc_drop'
    """
    from src.train import evaluate

    # First get baseline accuracy
    _, baseline_acc = evaluate(model, test_loader, device)

    results = []

    n_layers = len(model.transformer.layers)
    for l_idx in range(n_layers):
        for h_idx in range(model.n_heads):
            with ablate_head(model, l_idx, h_idx):
                _, abl_acc = evaluate(model, test_loader, device)

                results.append({
                    "layer": l_idx,
                    "head": h_idx,
                    "baseline_acc": baseline_acc,
                    "ablated_acc": abl_acc,
                    "acc_drop": baseline_acc - abl_acc
                })

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head", "baseline_acc", "ablated_acc", "acc_drop"])
        writer.writeheader()
        writer.writerows(results)

    return results
