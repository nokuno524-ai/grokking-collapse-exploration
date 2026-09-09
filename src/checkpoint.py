from typing import Tuple, Dict, Any, Optional
import torch
from pathlib import Path
import re
import json

def load_run(run_dir: Path, step: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
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
    res_path = run_dir / "results.json"
    if not cfg and res_path.exists():
        with res_path.open() as f:
            cfg = json.load(f).get("config", {})
    return sd, cfg
