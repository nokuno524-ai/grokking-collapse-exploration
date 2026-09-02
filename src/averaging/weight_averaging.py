import torch
from typing import Dict, List, Any

def load_checkpoint(filepath: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load model state dict from a checkpoint file.
    The expected format is a dict with at least 'model_state', which contains the weights.

    Args:
        filepath: Path to the checkpoint file (.pt).
        device: Device to load the tensors onto.

    Returns:
        The model state dict (dict of tensors).
    """
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        return ckpt["model_state"]
    # Fallback if the checkpoint is just a state dict
    return ckpt

def check_keys_match(sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor]) -> None:
    """
    Check if two state dicts have exactly the same keys and matching shapes.
    Raises ValueError if they do not match.
    """
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    if keys1 != keys2:
        missing_in_1 = keys2 - keys1
        missing_in_2 = keys1 - keys2
        raise ValueError(f"State dict keys do not match. "
                         f"Missing in first: {missing_in_1}. "
                         f"Missing in second: {missing_in_2}")

    for k in keys1:
        if sd1[k].shape != sd2[k].shape:
            raise ValueError(f"Shape mismatch for key '{k}': {sd1[k].shape} vs {sd2[k].shape}")

def interpolate_weights(
    sd_pre: Dict[str, torch.Tensor],
    sd_post: Dict[str, torch.Tensor],
    alpha: float
) -> Dict[str, torch.Tensor]:
    """
    Interpolate between two state dicts: alpha * sd_pre + (1 - alpha) * sd_post.

    Args:
        sd_pre: First state dict (e.g., early checkpoint).
        sd_post: Second state dict (e.g., late checkpoint).
        alpha: Interpolation factor in [0, 1].

    Returns:
        Interpolated state dict.
    """
    check_keys_match(sd_pre, sd_post)

    result = {}
    for k in sd_pre.keys():
        # Float cast to handle integer tensors like those sometimes used in buffers
        result[k] = alpha * sd_pre[k].float() + (1.0 - alpha) * sd_post[k].float()

        # Cast back to original dtype if it was an integer (e.g., num_batches_tracked)
        if not sd_pre[k].is_floating_point():
            result[k] = result[k].to(sd_pre[k].dtype)

    return result

def average_weights(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Compute the arithmetic mean of a list of state dicts.

    Args:
        state_dicts: List of state dicts.

    Returns:
        Averaged state dict.
    """
    if not state_dicts:
        raise ValueError("List of state dicts is empty.")

    n = len(state_dicts)
    if n == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    base_sd = state_dicts[0]
    for sd in state_dicts[1:]:
        check_keys_match(base_sd, sd)

    result = {}
    for k in base_sd.keys():
        # Accumulate in float to avoid overflow or precision loss
        accum = torch.zeros_like(base_sd[k], dtype=torch.float32)
        for sd in state_dicts:
            accum += sd[k].float()

        accum = accum / n

        if not base_sd[k].is_floating_point():
            accum = accum.to(base_sd[k].dtype)

        result[k] = accum

    return result
