import torch
import torch.nn as nn
from pathlib import Path
from src.model import ModularArithmeticTransformer

def load_run(run_dir, step=None, strict=True):
    """
    Load a checkpoint for the model from a run directory.
    If step is None, the last checkpoint is loaded.

    Args:
        run_dir: Path to the run directory
        step: Checkpoint step to load. Defaults to None (loads last checkpoint).
        strict: Whether to enforce strict key matching in state_dict. Default is True.

    Returns:
        model: The loaded model, in eval() mode.
        step: The step of the loaded checkpoint.
    """
    run_dir = Path(run_dir)
    if step is None:
        ckpts = list(run_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        step = max([int(p.stem.split("_")[1]) for p in ckpts])

    ckpt_path = run_dir / f"checkpoint_{step}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # We may need to get model config from results.json, but for now we assume defaults
    # Let's try to infer prime and other args if we can, but since this is toy model we assume ModularArithmeticTransformer()
    # Actually, it's safer to just initialize and load.

    # The config isn't in checkpoint unfortunately, let's try reading config from results.json if we need to.
    import json
    results_path = run_dir / "results.json"
    kwargs = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
            if "config" in data:
                cfg = data["config"]
                # Map keys from trainconfig to model config
                # TrainConfig has prime, d_model, n_heads, d_ff, n_layers, dropout(None but let's assume 0.0)
                for k in ["prime", "d_model", "n_heads", "d_ff", "n_layers"]:
                    if k in cfg:
                        kwargs[k] = cfg[k]

    model = ModularArithmeticTransformer(**kwargs)

    try:
        model.load_state_dict(ckpt, strict=strict)
    except Exception as e:
        if strict:
            raise e
        else:
            print(f"Warning: strict=False. Ignoring missing keys in checkpoint {ckpt_path}: {e}")

    model.eval()
    return model, step
