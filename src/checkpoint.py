import re
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
import warnings

def load_run(run_dir: Path, step: Optional[int] = None) -> Tuple[Dict, dict]:
    """Return (state_dict, config) for the given run.
    If step is None, picks the largest checkpoint.

    (We will just parse the config and fallback to results.json if needed)."""
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
        try:
            with res_path.open() as f:
                cfg = json.load(f).get("config", {})
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    return sd, cfg
