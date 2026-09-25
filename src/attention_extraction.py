import contextlib
import numpy as np
import torch
import torch.nn as nn
from typing import Iterator, Dict, Any, List

@contextlib.contextmanager
def extract_attention(model: nn.Module) -> Iterator[Dict[str, List[np.ndarray]]]:
    """
    Context manager that temporarily patches nn.MultiheadAttention.forward
    to extract and capture attention weights, specifically disabling PyTorch
    2.0+ SDPA fast-paths.

    Yields:
        A dictionary where keys are module names and values are lists of
        extracted attention weight matrices (as numpy arrays) for each forward pass.
    """
    extracted_weights: Dict[str, List[np.ndarray]] = {}
    original_forwards = {}

    # We must disable SDPA fastpath for PyTorch 2.0+ to ensure we get attention weights out.
    # We also ensure the model is in eval mode, as some hooks/behaviors differ in train mode.
    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    was_training = model.training
    model.eval()

    def get_forward_hook(name: str, original_forward):
        def patched_forward(self, *args, **kwargs):
            import inspect
            sig = inspect.signature(original_forward)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound.arguments['need_weights'] = True
            bound.arguments['average_attn_weights'] = False

            output, attn_weights = original_forward(*bound.args, **bound.kwargs)

            if name not in extracted_weights:
                extracted_weights[name] = []

            # Detach, move to CPU, and convert to numpy
            extracted_weights[name].append(attn_weights.detach().cpu().numpy())

            return output, attn_weights
        return patched_forward

    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                original_forwards[name] = module.forward
                # Need to use __get__ to bind the patched method to the instance
                module.forward = get_forward_hook(name, module.forward).__get__(module, type(module))

        yield extracted_weights

    finally:
        # Restore original forwards
        for name, module in model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]

        # Restore fastpath and train state
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)
        if was_training:
            model.train()


from pathlib import Path
from src.checkpoint import load_run
from src.data import generate_modular_arithmetic

def compute_entropy(attn_weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute row-wise Shannon entropy of attention matrices.

    Args:
        attn_weights: Array of shape (batch, heads, seq_len, seq_len)

    Returns:
        Entropy array of shape (batch, heads, seq_len)
    """
    # Clip to avoid log(0)
    p = np.clip(attn_weights, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=-1)
    return entropy

def compute_diagonal_dominance(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal dominance (trace / sequence length) of the attention matrices.
    Since this is (a+b) mod p (seq_len=2), the diagonal is a->a and b->b.

    Args:
        attn_weights: Array of shape (batch, heads, seq_len, seq_len)

    Returns:
        Diagonal dominance array of shape (batch, heads)
    """
    batch_size, heads, seq_len, _ = attn_weights.shape
    diagonals = np.diagonal(attn_weights, axis1=-2, axis2=-1)
    return np.sum(diagonals, axis=-1) / seq_len

def run_attention_extraction(
    run_dir: str | Path,
    step: int | None = None,
    batch_size: int = 512,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load a checkpoint, run an evaluation batch, and extract attention matrices.

    Args:
        run_dir: Path to the run directory
        step: Specific checkpoint step to load (latest if None)
        batch_size: Batch size for extraction
        device: Device to run extraction on

    Returns:
        Dictionary with extraction results and metadata
    """
    run_dir = Path(run_dir)
    model, cfg = load_run(run_dir, step=step, strict=False, device=device)

    from src.data import DatasetConfig
    prime = cfg.get("prime", 59)
    # Generate fixed evaluation batch for deterministic extraction
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(DatasetConfig(prime=prime))

    # Combine all data for extraction
    all_in = torch.cat([train_in, test_in], dim=0)
    all_tgt = torch.cat([train_tgt, test_tgt], dim=0)

    # Use fixed seed for deterministic batch
    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_in), min(batch_size, len(all_in)), replace=False)

    batch = all_in[indices].to(device)
    targets = all_tgt[indices].to(device)

    with extract_attention(model) as extracted:
        with torch.no_grad():
            _ = model(batch)

    # Assuming single forward pass, grab the first element for each module
    attn_dict = {name: val[0] for name, val in extracted.items()}

    # Return structure with metadata
    return {
        "step": step if step is not None else -1, # -1 for latest
        "run_dir": str(run_dir),
        "seed": cfg.get("seed", -1),
        "prime": prime,
        "batch_inputs": batch.cpu().numpy(),
        "batch_targets": targets.cpu().numpy(),
        "attention_weights": attn_dict,
        "config": cfg
    }

def cache_attention_extraction(
    run_dir: str | Path,
    output_path: str | Path,
    step: int | None = None,
    batch_size: int = 512,
    device: str = "cpu"
) -> None:
    """
    Extract attention and save it to an .npz file.

    Args:
        run_dir: Path to the run directory
        output_path: Path to save the .npz file
        step: Specific checkpoint step
        batch_size: Batch size
        device: Device
    """
    res = run_attention_extraction(run_dir, step, batch_size, device)

    save_dict = {
        "step": res["step"],
        "run_dir": res["run_dir"],
        "seed": res["seed"],
        "prime": res["prime"],
        "batch_inputs": res["batch_inputs"],
        "batch_targets": res["batch_targets"]
    }

    # Flatten attention weights into main dict
    for name, weights in res["attention_weights"].items():
        save_dict[f"attn_{name}"] = weights

    np.savez(output_path, **save_dict)
