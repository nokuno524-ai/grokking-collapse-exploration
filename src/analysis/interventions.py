import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def ablate_head(model: nn.Module, head_idx: int) -> nn.Module:
    """
    Ablate a specific attention head by zeroing its output weights.
    Operates on a 1-layer ModularArithmeticTransformer.

    Args:
        model: The model to modify (will be modified in-place, but returned for convenience).
        head_idx: The index of the head to ablate.

    Returns:
        The modified model.
    """
    n_heads = model.n_heads
    d_model = model.d_model
    head_dim = d_model // n_heads

    with torch.no_grad():
        out_proj = model.transformer.layers[0].self_attn.out_proj
        # Zero out the weights corresponding to the specific head
        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim
        out_proj.weight[:, start_idx:end_idx] = 0.0

    return model

def counterfactual_patch(model_a: nn.Module, model_b: nn.Module, layer: int, position: int = None) -> nn.Module:
    """
    Patch the activation from model_a into model_b at a specific layer.
    (Because we use nn.TransformerEncoderLayer which doesn't expose internal head outputs easily,
     we'll patch the whole layer output. For position, if specified, patch only that seq pos).

    We modify model_b in place by attaching a forward hook that fetches from model_a.
    """

    cache = {}

    def cache_hook(module, args, output):
        cache['act'] = output.detach()
        return output

    def patch_hook(module, args, output):
        patched_output = output.clone()
        if position is not None:
            patched_output[:, position, :] = cache['act'][:, position, :]
        else:
            patched_output = cache['act']
        return patched_output

    model_a.transformer.layers[layer].register_forward_hook(cache_hook)
    model_b.transformer.layers[layer].register_forward_hook(patch_hook)

    return model_b

def run_intervention_suite(model: nn.Module, dataset: torch.utils.data.Dataset) -> Dict[str, float]:
    """
    Systematically ablate heads and report the change in loss.

    Args:
        model: The base model (should be a copy or we need to restore it).
        dataset: Dataset to evaluate on.

    Returns:
        Dict mapping 'head_{idx}' to the increase in loss after ablation.
    """
    import copy

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)
    criterion = nn.CrossEntropyLoss()

    def evaluate(m):
        m.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                logits = m(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
        return total_loss / len(dataset)

    base_loss = evaluate(model)
    results = {}

    for head_idx in range(model.n_heads):
        # Create a fresh copy for each ablation
        model_copy = copy.deepcopy(model)
        ablate_head(model_copy, head_idx)
        ablated_loss = evaluate(model_copy)

        # We report the *increase* in loss (higher = more important)
        results[f'head_{head_idx}_loss_diff'] = ablated_loss - base_loss

    return results


def ablate_mlp_neurons(model: nn.Module, layer_idx: int, neuron_indices: List[int]) -> nn.Module:
    """
    Ablate specific neurons in the MLP layer of the transformer block.

    Args:
        model: The model to modify (will be modified in-place).
        layer_idx: The layer index to ablate in.
        neuron_indices: The indices of the neurons to zero out in the intermediate dimension.

    Returns:
        The modified model.
    """
    with torch.no_grad():
        linear2 = model.transformer.layers[layer_idx].linear2
        # Zero out the input weights corresponding to these neurons
        # linear2 has shape (d_model, dim_feedforward)
        for idx in neuron_indices:
            linear2.weight[:, idx] = 0.0

    return model
