import re
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
from src.model import ModularArithmeticTransformer

def load_run(run_dir: Path, step: Optional[int] = None, strict: bool = True, device: Optional[torch.device] = None) -> Tuple[nn.Module, dict]:
    """
    Centralized checkpoint loading.
    Returns (model, config) for the given run.
    If step is None, picks the largest checkpoint.
    """
    if device is None:
        device = torch.device("cpu")

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
        sd = ckpt
        cfg = {}

    res_path = run_dir / "results.json"
    if not cfg and res_path.exists():
        with res_path.open() as f:
            try:
                cfg = json.load(f).get("config", {})
            except json.JSONDecodeError:
                pass

    # Build model
    prime = cfg.get("prime", 59)
    d_model = cfg.get("d_model", 128)
    n_heads = cfg.get("n_heads", 4)
    d_ff = cfg.get("d_ff", 512)
    n_layers = cfg.get("n_layers", 1)

    model = ModularArithmeticTransformer(
        prime=prime,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
    )

    # Load state dict
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if not strict:
        if missing:
            print(f"Warning: Missing keys when loading {chosen}: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading {chosen}: {unexpected}")

    model = model.to(device)
    model.eval() # Ensure evaluation mode
    return model, cfg
