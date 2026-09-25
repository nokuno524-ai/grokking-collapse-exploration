import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

import torch
import torch.nn as nn

from src.model import ModularArithmeticTransformer

def load_run(run_dir: str | Path, step: Optional[int] = None, strict: bool = True, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """
    Load a model and its configuration from a run directory.

    Args:
        run_dir: Path to the run directory containing checkpoint_*.pt and results.json
        step: Specific step to load. If None, loads the latest checkpoint.
        strict: Whether to enforce strict state dict loading. If False, prints warnings instead.
        device: Device to load the model on.

    Returns:
        A tuple of (model, config)
    """
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir}")

    chosen: Optional[Path] = None
    if step is not None:
        for p in ckpts:
            if int(re.findall(r"\d+", p.name)[-1]) == step:
                chosen = p
                break
        if chosen is None:
            raise FileNotFoundError(f"no checkpoint_{step}.pt in {run_dir}")
    else:
        chosen = ckpts[-1]

    ckpt = torch.load(chosen, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    else:
        # Fallback if checkpoint is just the state dict
        sd = ckpt
        cfg = {}
        # Try to load config from results.json
        results_path = run_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
                cfg = res.get("config", {})

    model = ModularArithmeticTransformer(
        prime=cfg.get("prime", 59),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        d_ff=cfg.get("d_ff", 512),
        n_layers=cfg.get("n_layers", 1),
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if strict:
        if missing or unexpected:
            raise RuntimeError(f"Error loading state_dict. Missing: {missing}, Unexpected: {unexpected}")
    else:
        if missing:
            warnings.warn(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys when loading state_dict: {unexpected}")

    model.to(device)
    model.eval()

    return model, cfg
