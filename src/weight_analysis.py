import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.model import ModularArithmeticTransformer
from src.stats_utils import detect_phase_transition

def track_weight_norm_distribution(
    model: ModularArithmeticTransformer
) -> Dict[str, float]:
    """
    Track layer-wise weight norm distribution.

    Args:
        model: Model to analyze

    Returns:
        Dict mapping layer names to their L2 norm
    """
    norms = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            norms[name] = param.norm().item()
    return norms


def get_svd_spectrum(
    model: ModularArithmeticTransformer,
    layer_name: str = "token_embed.weight"
) -> np.ndarray:
    """
    Analyze the Singular Value spectrum of a weight matrix.

    Args:
        model: Model to analyze
        layer_name: Name of the parameter to compute SVD for

    Returns:
        Array of singular values
    """
    for name, param in model.named_parameters():
        if name == layer_name:
            W = param.detach()
            if W.ndim > 2:
                W = W.reshape(W.size(0), -1)
            # Compute SVD
            s = torch.linalg.svdvals(W)
            return s.cpu().numpy()

    raise ValueError(f"Layer {layer_name} not found in model")


def analyze_weight_trajectory(
    checkpoints: List[Path],
    device: torch.device
) -> Dict[str, Any]:
    """
    Analyze weight structure evolution over training and detect phase transitions.

    Args:
        checkpoints: List of checkpoint paths
        device: Torch device

    Returns:
        Dictionary with tracking history and detected transition points
    """
    if not checkpoints:
        return {}

    ckpt = torch.load(checkpoints[0], map_location=device)
    cfg = ckpt.get("config", {})

    model = ModularArithmeticTransformer(
        prime=cfg.get("prime", 59),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        d_ff=cfg.get("d_ff", 512),
        n_layers=cfg.get("n_layers", 1),
    ).to(device)

    history = {
        "steps": [],
        "norms": {},
        "svd_embed": [],
        "svd_out": [],
        "effective_rank_embed": []
    }

    for cp in checkpoints:
        ckpt = torch.load(cp, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        step = ckpt.get("step", 0)
        history["steps"].append(step)

        # Norms
        norms = track_weight_norm_distribution(model)
        for k, v in norms.items():
            if k not in history["norms"]:
                history["norms"][k] = []
            history["norms"][k].append(v)

        # SVD
        s_embed = get_svd_spectrum(model, "token_embed.weight")
        history["svd_embed"].append(s_embed)

        # Effective rank
        s_norm = s_embed / s_embed.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
        history["effective_rank_embed"].append(np.exp(entropy))

        s_out = get_svd_spectrum(model, "output_head.weight")
        history["svd_out"].append(s_out)

    # Detect phase transitions
    transitions = {}

    # Transition in embedding rank
    rank_series = np.array(history["effective_rank_embed"])
    idx = detect_phase_transition(rank_series)
    if idx >= 0:
        transitions["rank_transition_step"] = history["steps"][idx]

    # Transition in norms
    for k, v in history["norms"].items():
        idx = detect_phase_transition(v)
        if idx >= 0:
            transitions[f"{k}_transition_step"] = history["steps"][idx]

    history["transitions"] = transitions
    return history


def correlate_weight_grokking(
    results_dir: str,
    device: torch.device
) -> Dict[str, Dict[str, int]]:
    """
    Correlate weight structure phase transitions with grokking timing.
    """
    from pathlib import Path
    import json

    results_path = Path(results_dir)
    conditions = [d for d in results_path.iterdir() if d.is_dir()]

    correlations = {}

    for condition_dir in conditions:
        checkpoints = sorted(condition_dir.glob("checkpoint_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            continue

        # Get grokking step from results.json if available
        results_file = condition_dir / "results.json"
        grok_step = -1
        if results_file.exists():
            with open(results_file, "r") as f:
                data = json.load(f)
                grok_step = data.get("grokking_step", -1)

        # Analyze weight trajectory
        history = analyze_weight_trajectory(checkpoints, device)
        transitions = history.get("transitions", {})

        correlations[condition_dir.name] = {
            "grok_step": grok_step,
            "rank_transition": transitions.get("rank_transition_step", -1),
            "embed_norm_transition": transitions.get("token_embed.weight_transition_step", -1),
            "out_norm_transition": transitions.get("output_head.weight_transition_step", -1)
        }

    return correlations
