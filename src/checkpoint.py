import torch
import os
import glob
from pathlib import Path
from typing import Optional, Union

def load_run(run_dir: Union[str, Path], step: Optional[int] = None, strict: bool = True) -> torch.nn.Module:
    """
    Loads a model checkpoint from a run directory.

    If step is None, loads the final model (`model.pt`) or the latest step checkpoint.
    Applies `model.eval()` before returning to avoid mode leaks during analysis.

    Args:
        run_dir: Path to the run directory containing the config.json and checkpoints.
        step: Specific checkpoint step to load (e.g. 1000 for `checkpoint_1000.pt`).
        strict: Whether to strictly enforce state_dict matching.

    Returns:
        The instantiated and loaded ModularArithmeticTransformer model in eval mode.
    """
    from src.model import ModularArithmeticTransformer
    import json

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        model = ModularArithmeticTransformer(
            prime=config_dict.get("prime", 59),
            d_model=config_dict.get("d_model", 128),
            n_heads=config_dict.get("n_heads", 4),
            d_ff=config_dict.get("d_ff", 512),
            n_layers=config_dict.get("n_layers", 1),
            dropout=config_dict.get("dropout", 0.0),
        )
    else:
        model = ModularArithmeticTransformer()

    if step is not None:
        ckpt_path = run_dir / f"checkpoint_{step}.pt"
    else:
        # try to find final model
        ckpt_path = run_dir / "model.pt"
        if not ckpt_path.exists():
            # Find the latest checkpoint
            ckpts = glob.glob(str(run_dir / "checkpoint_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {run_dir}")

            def parse_step(p: str) -> int:
                try:
                    return int(Path(p).stem.split("_")[1])
                except ValueError:
                    return -1
            ckpts.sort(key=parse_step)
            ckpt_path = Path(ckpts[-1])

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if not strict:
            print(f"Warning: missing/unexpected keys: {e}")
        else:
            raise e

    model.eval()
    return model
