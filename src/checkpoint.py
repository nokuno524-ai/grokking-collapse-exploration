import torch
from pathlib import Path
from typing import Tuple, Dict, Optional
import re

def load_run(run_dir: Path, step: Optional[int] = None, strict: bool = True) -> Tuple[Dict, dict]:
    """Return (state_dict, config) for the given run.
    If step is None, picks the largest checkpoint."""
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

    if not strict:
        import warnings
        warnings.warn("strict=False passed to load_run, mismatching keys will be ignored")

    return sd, cfg
