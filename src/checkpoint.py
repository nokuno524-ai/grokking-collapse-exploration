import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import torch

def load_run_config(run_dir: Path) -> Optional[dict]:
    """
    Load config from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        A dictionary containing the configuration, or None if not found.
    """
    res_path = run_dir / "results.json"
    if not res_path.exists():
        return None
    try:
        with res_path.open() as f:
            data = json.load(f)
        return data.get("config", data)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def list_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """
    List all checkpoints in a directory, sorted by step.

    Args:
        run_dir: Path to the run directory.

    Returns:
        A list of tuples (step, path_to_checkpoint), sorted by step.
    """
    ckpts = []
    for p in run_dir.glob("checkpoint_*.pt"):
        match = re.search(r"checkpoint_(\d+)\.pt", p.name)
        if match:
            ckpts.append((int(match.group(1)), p))
    return sorted(ckpts, key=lambda x: x[0])

def load_checkpoint(
    run_dir: Path,
    step: Optional[int] = None,
    strict_config_match: Optional[dict] = None
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    Load a checkpoint from a run directory.
    If step is None, loads the largest (latest) checkpoint.

    Args:
        run_dir: Path to the run directory.
        step: Specific step to load. If None, loads the latest.
        strict_config_match: Optional config to compare against for architecture shape mismatch.

    Returns:
        A tuple of (state_dict, config).
    """
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir}")

    chosen_path = None
    if step is not None:
        for s, p in ckpts:
            if s == step:
                chosen_path = p
                break
        if chosen_path is None:
            raise FileNotFoundError(f"no checkpoint_{step}.pt in {run_dir}")
    else:
        chosen_path = ckpts[-1][1]

    try:
        ckpt = torch.load(chosen_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {chosen_path}: {e}")

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    else:
        sd = ckpt
        cfg = {}

    file_cfg = load_run_config(run_dir)
    if file_cfg:
        cfg = file_cfg

    if strict_config_match is not None:
        # Check architecture matches
        for key in ["prime", "d_model", "n_heads", "d_ff", "n_layers"]:
            if str(cfg.get(key, "")) != str(strict_config_match.get(key, "")) and key in cfg and key in strict_config_match:
                raise ValueError(
                    f"architecture mismatch on {key}: pure={strict_config_match.get(key)} "
                    f"vs contam={cfg.get(key)}"
                )

    return sd, cfg
