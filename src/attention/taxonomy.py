import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Checkpoint loading is actually not centralized in src/checkpoint.py
# Let's just use torch.load
from src.attention.extraction import extract_attention, compute_head_features
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def cluster_heads(features: Dict[str, np.ndarray], threshold: float = 0.3) -> List[str]:
    """
    Cluster heads into classes using a simple threshold rule based on the extracted features.

    Classes:
    - 'positional': offset_1_mass or offset_2_mass > threshold
    - 'copy': copy_score > threshold
    - 'induction': induction_score > threshold
    - 'other': none of the above

    Args:
        features: Dictionary mapping feature name to (batch, n_heads) arrays.
        threshold: The threshold above which a head is considered to belong to a class.

    Returns:
        List of class names for each head, length n_heads.
    """
    # Average features across the batch
    avg_features = {k: v.mean(axis=0) for k, v in features.items()}

    n_heads = list(avg_features.values())[0].shape[0]
    classes = []

    for h in range(n_heads):
        if avg_features.get('induction_score', np.zeros(n_heads))[h] > threshold:
            classes.append('induction')
        elif avg_features.get('copy_score', np.zeros(n_heads))[h] > threshold:
            classes.append('copy')
        elif avg_features.get('offset_1_mass', np.zeros(n_heads))[h] > threshold or \
             avg_features.get('offset_2_mass', np.zeros(n_heads))[h] > threshold:
            classes.append('positional')
        else:
            classes.append('other')

    return classes

def build_atlas(checkpoint_paths: List[Path], data_config: DatasetConfig, out_path: Optional[Path] = None, threshold: float = 0.3) -> Dict[str, Any]:
    """
    Load checkpoints, extract attention features on validation data,
    and save the head taxonomy atlas to JSON.

    Args:
        checkpoint_paths: List of paths to checkpoint files.
        data_config: DatasetConfig used to generate calibration data.
        out_path: Optional path to save the resulting JSON atlas.
        threshold: Threshold for clustering rules.

    Returns:
        Dictionary representing the atlas.
    """
    # Generate some validation data
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)
    # We will just use test_in (batch, seq_len=2) for feature extraction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = test_in.to(device)

    atlas = {}

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt['config']

        # Instantiate model
        model = ModularArithmeticTransformer(
            prime=cfg.get('prime', 59),
            d_model=cfg.get('d_model', 128),
            n_heads=cfg.get('n_heads', 4),
            d_ff=cfg.get('d_ff', 512),
            n_layers=cfg.get('n_layers', 1),
        ).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        step = ckpt.get('step', str(ckpt_path.name))

        # Extract attention weights
        with extract_attention(model) as attentions:
            # We don't care about the output
            _ = model(tokens)

        if not attentions:
            atlas[str(step)] = ["unknown"] * model.n_heads
            continue

        # attentions is a list of (batch, n_heads, seq_len, seq_len)
        # the model only has 1 layer so it's a list of 1 element
        attn_weights = attentions[0]

        # Compute features
        features = compute_head_features(attn_weights, tokens.cpu().numpy())

        # Cluster
        head_classes = cluster_heads(features, threshold=threshold)

        atlas[str(step)] = head_classes

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(atlas, f, indent=2)

    return atlas
