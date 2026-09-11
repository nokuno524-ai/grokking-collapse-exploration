import torch
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch.nn as nn

def load_run(run_dir: Path | str, step: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a training checkpoint (state dict and config) for a specific step.
    If step is None, loads the checkpoint with the highest step number.

    Args:
        run_dir: Path to the directory containing checkpoint files.
        step: Specific step to load. Defaults to None (latest).

    Returns:
        A tuple of (model_state_dict, config_dict).

    Raises:
        FileNotFoundError: If the directory or the specified checkpoint doesn't exist.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_*.pt files found in {run_dir}")

    chosen: Optional[Path] = None
    if step is not None:
        for p in ckpts:
            if int(re.findall(r"\d+", p.name)[-1]) == step:
                chosen = p
                break
        if chosen is None:
            raise FileNotFoundError(f"No checkpoint found for step {step} in {run_dir}")
    else:
        chosen = ckpts[-1]

    ckpt = torch.load(chosen, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    elif isinstance(ckpt, dict) and any(k.startswith("transformer") for k in ckpt.keys()):
        # Legacy raw state dict
        sd = ckpt
        cfg = {}
    else:
        # Fallback
        sd = ckpt
        cfg = {}

    return sd, cfg

def load_model_from_run(model: nn.Module, run_dir: Path | str, step: Optional[int] = None, strict: bool = True) -> Dict[str, Any]:
    """
    Loads a checkpoint into a model and sets it to eval mode.

    Args:
        model: The model to load weights into.
        run_dir: Directory containing checkpoints.
        step: Step to load (None for latest).
        strict: Whether to enforce strict key matching.

    Returns:
        The config dictionary associated with the loaded run.
    """
    sd, cfg = load_run(run_dir, step)

    if strict:
        model.load_state_dict(sd, strict=True)
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"Warning: load_state_dict with strict=False encountered Missing keys: {missing}, Unexpected keys: {unexpected}")

    model.eval()
    return cfg