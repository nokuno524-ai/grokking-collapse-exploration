import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable
import copy

from src.model import ModularArithmeticTransformer

def activation_patching(
    model: ModularArithmeticTransformer,
    clean_inputs: torch.Tensor,
    corrupted_inputs: torch.Tensor,
    patch_layer: str,
    metric_fn: Callable[[torch.Tensor], float]
) -> float:
    """
    Perform activation patching to identify important components.
    Runs a forward pass on corrupted_inputs, but replaces the activation of
    patch_layer with the activation from clean_inputs.

    Args:
        model: ModularArithmeticTransformer
        clean_inputs: Inputs that yield the desired output
        corrupted_inputs: Inputs that yield an undesired output
        patch_layer: Name of the layer to patch ("embed", "attn", "ffn")
        metric_fn: Function to compute a metric (e.g., probability of correct answer) from logits

    Returns:
        The metric value after patching.
    """
    device = clean_inputs.device
    batch_size = clean_inputs.shape[0]

    # Run clean forward pass and cache activations
    cache = {}

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                cache[name] = output[0].detach()
            else:
                cache[name] = output.detach()
        return hook

    # Register hooks for clean pass
    hooks = []
    if patch_layer == "embed":
        hooks.append(model.token_embed.register_forward_hook(get_activation("embed")))
    elif patch_layer == "attn":
        hooks.append(model.transformer.layers[0].self_attn.register_forward_hook(get_activation("attn")))
    elif patch_layer == "ffn":
        hooks.append(model.transformer.layers[0].linear2.register_forward_hook(get_activation("ffn")))

    with torch.no_grad():
        model(clean_inputs)

    for hook in hooks:
        hook.remove()

    # Run corrupted forward pass with patching
    def patch_activation(name):
        def hook(model, input, output):
            # If output is a tuple (like from self_attn), patch the first element
            if isinstance(output, tuple):
                return (cache[name],) + output[1:]
            return cache[name]
        return hook

    patch_hooks = []
    if patch_layer == "embed":
        patch_hooks.append(model.token_embed.register_forward_hook(patch_activation("embed")))
    elif patch_layer == "attn":
        patch_hooks.append(model.transformer.layers[0].self_attn.register_forward_hook(patch_activation("attn")))
    elif patch_layer == "ffn":
        patch_hooks.append(model.transformer.layers[0].linear2.register_forward_hook(patch_activation("ffn")))

    with torch.no_grad():
        patched_logits = model(corrupted_inputs)

    for hook in patch_hooks:
        hook.remove()

    return metric_fn(patched_logits)


def trace_information_flow(
    model: ModularArithmeticTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Trace information flow by systematically patching layers.

    Returns a dictionary of patching effects for each major component.
    """
    # Create corrupted inputs by adding a random offset
    device = inputs.device
    prime = model.prime
    offset = torch.randint(1, prime, inputs.shape, device=device)
    corrupted_inputs = (inputs + offset) % prime

    def acc_metric(logits):
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()

    # Baseline corrupted accuracy
    with torch.no_grad():
        corrupted_logits = model(corrupted_inputs)
    baseline_corrupted_acc = acc_metric(corrupted_logits)

    # Baseline clean accuracy
    with torch.no_grad():
        clean_logits = model(inputs)
    baseline_clean_acc = acc_metric(clean_logits)

    effects = {
        "baseline_clean": baseline_clean_acc,
        "baseline_corrupted": baseline_corrupted_acc
    }

    for layer in ["embed", "attn", "ffn"]:
        patched_acc = activation_patching(model, inputs, corrupted_inputs, layer, acc_metric)
        # Calculate rescue effect: how much patching this layer recovers the clean performance
        if baseline_clean_acc > baseline_corrupted_acc:
            rescue = (patched_acc - baseline_corrupted_acc) / (baseline_clean_acc - baseline_corrupted_acc)
        else:
            rescue = 0.0
        effects[layer] = rescue

    return effects


def find_minimal_grokking_circuit(
    model: ModularArithmeticTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    acc_threshold: float = 0.90
) -> List[int]:
    """
    Greedy ablation to find minimal subgraphs (attention heads) that maintain accuracy.
    Returns a list of head indices that form the minimal circuit.
    """
    from src.train import evaluate

    n_heads = model.n_heads
    kept_heads = list(range(n_heads))

    _, baseline_acc = evaluate(model, dataloader, device)
    if baseline_acc < acc_threshold:
        return kept_heads  # Model already below threshold, keep everything

    while len(kept_heads) > 1:
        best_acc = -1.0
        head_to_remove = -1

        # Try removing each currently kept head
        for h in kept_heads:
            test_model = copy.deepcopy(model)
            layer = test_model.transformer.layers[0]
            d_model = layer.self_attn.embed_dim
            head_dim = d_model // n_heads

            # Zero out all heads EXCEPT the ones we're testing keeping
            heads_to_keep = [k for k in kept_heads if k != h]

            with torch.no_grad():
                # Zero everything first
                layer.self_attn.out_proj.weight.fill_(0.0)
                # Restore the ones we want to keep
                for keep_h in heads_to_keep:
                    start_idx = keep_h * head_dim
                    end_idx = (keep_h + 1) * head_dim
                    layer.self_attn.out_proj.weight[:, start_idx:end_idx] = \
                        model.transformer.layers[0].self_attn.out_proj.weight[:, start_idx:end_idx]

            _, acc = evaluate(test_model, dataloader, device)

            if acc > best_acc:
                best_acc = acc
                head_to_remove = h

        # If removing the best candidate still keeps us above threshold, remove it
        if best_acc >= acc_threshold:
            kept_heads.remove(head_to_remove)
        else:
            break  # Cannot remove any more heads without dropping below threshold

    return kept_heads


def compare_circuit_complexity(
    results_dir: str,
    device: torch.device
) -> Dict[str, int]:
    """
    Compare circuit complexity (number of required heads) across collapse conditions.
    """
    import torch.utils.data as data
    from src.data import generate_modular_arithmetic, DatasetConfig
    from pathlib import Path

    results_path = Path(results_dir)
    conditions = [d for d in results_path.iterdir() if d.is_dir()]

    complexities = {}

    for condition_dir in conditions:
        # Find the final checkpoint
        checkpoints = sorted(condition_dir.glob("checkpoint_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            continue

        final_cp = checkpoints[-1]
        ckpt = torch.load(final_cp, map_location=device)
        cfg = ckpt.get("config", {})

        model = ModularArithmeticTransformer(
            prime=cfg.get("prime", 59),
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            d_ff=cfg.get("d_ff", 512),
            n_layers=cfg.get("n_layers", 1),
        ).to(device)

        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Create a small dataset for evaluation
        data_cfg = DatasetConfig(
            prime=cfg.get("prime", 59),
            train_fraction=cfg.get("train_fraction", 0.3),
            seed=cfg.get("seed", 42)
        )
        _, _, test_in, test_tgt = generate_modular_arithmetic(data_cfg)
        test_loader = data.DataLoader(data.TensorDataset(test_in, test_tgt), batch_size=512)

        # Find minimal circuit
        minimal_heads = find_minimal_grokking_circuit(model, test_loader, device)
        complexities[condition_dir.name] = len(minimal_heads)

    return complexities
