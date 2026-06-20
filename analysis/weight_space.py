"""
Weight Space Analysis.
Computes weight norm evolution per layer, singular value spectra of weight matrices,
effective rank (participation ratio), cosine distance between checkpoints,
and CKA comparison across collapse conditions.
"""

import torch
import numpy as np
import math
from pathlib import Path

def compute_effective_rank(singular_values: torch.Tensor, epsilon: float = 1e-10) -> float:
    """
    Compute the effective rank (Shannon entropy) from singular values.
    """
    if singular_values.numel() == 0:
        return 0.0
    s = singular_values / (singular_values.sum() + epsilon)
    entropy = -(s * torch.log(s + epsilon)).sum()
    return torch.exp(entropy).item()

def compute_layer_norms(model_state: dict) -> dict:
    """
    Compute L2 norm for each weight matrix in the state dict.
    """
    norms = {}
    for name, param in model_state.items():
        if 'weight' in name and param.dim() >= 2:
            norms[name] = param.norm().item()
    return norms

def compute_cosine_distance(state1: dict, state2: dict) -> float:
    """
    Compute global cosine distance between two model states.
    Cosine distance = 1 - cosine_similarity.
    """
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for name in state1.keys():
        if name in state2:
            p1 = state1[name].view(-1)
            p2 = state2[name].view(-1)
            dot_product += (p1 * p2).sum().item()
            norm1 += (p1 * p1).sum().item()
            norm2 += (p2 * p2).sum().item()

    norm1 = math.sqrt(norm1)
    norm2 = math.sqrt(norm2)

    if norm1 == 0 or norm2 == 0:
        return 1.0

    sim = dot_product / (norm1 * norm2)
    # Clip for floating point precision
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim

def linear_cka(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA) between two Gram matrices.
    """
    def center_gram(g):
        n = g.shape[0]
        h = np.eye(n) - np.ones((n, n)) / n
        return h.dot(g).dot(h)

    g_x = center_gram(gram_x)
    g_y = center_gram(gram_y)

    scaled_hsic = np.trace(g_x.dot(g_y))
    norm_x = np.linalg.norm(g_x, 'fro')
    norm_y = np.linalg.norm(g_y, 'fro')

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return scaled_hsic / (norm_x * norm_y)

def compute_activation_cka(acts1: torch.Tensor, acts2: torch.Tensor) -> float:
    """
    Compute CKA between two sets of activations of shape (batch, features).
    """
    x = acts1.view(acts1.size(0), -1).cpu().numpy()
    y = acts2.view(acts2.size(0), -1).cpu().numpy()

    gram_x = x.dot(x.T)
    gram_y = y.dot(y.T)

    return linear_cka(gram_x, gram_y)

def analyze_weight_space(checkpoint_dir: str | Path):
    """
    Analyze weight norms, effective ranks, and cosine distances across checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(
        [p for p in checkpoint_dir.glob("checkpoint_*.pt")],
        key=lambda p: int(p.stem.split('_')[1])
    )

    metrics = {
        'steps': [],
        'layer_norms': {},
        'effective_ranks': {},
        'cosine_distance_to_init': [],
        'cosine_distance_to_prev': [],
    }

    init_state = None
    prev_state = None

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split('_')[1])
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model_state = ckpt['model_state']
        except Exception:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model_state = ckpt['model_state']
            except Exception as e:
                print(f"Skipping {ckpt_path}: {e}")
                continue

        metrics['steps'].append(step)

        # Norms
        norms = compute_layer_norms(model_state)
        for k, v in norms.items():
            if k not in metrics['layer_norms']:
                metrics['layer_norms'][k] = []
            metrics['layer_norms'][k].append(v)

        # Ranks
        ranks = {}
        for name, param in model_state.items():
            if 'weight' in name and param.dim() >= 2:
                s = torch.linalg.svdvals(param.float())
                ranks[name] = compute_effective_rank(s)

        for k, v in ranks.items():
            if k not in metrics['effective_ranks']:
                metrics['effective_ranks'][k] = []
            metrics['effective_ranks'][k].append(v)

        # Distances
        if init_state is None:
            init_state = model_state
            metrics['cosine_distance_to_init'].append(0.0)
            metrics['cosine_distance_to_prev'].append(0.0)
        else:
            dist_init = compute_cosine_distance(init_state, model_state)
            dist_prev = compute_cosine_distance(prev_state, model_state)
            metrics['cosine_distance_to_init'].append(dist_init)
            metrics['cosine_distance_to_prev'].append(dist_prev)

        prev_state = model_state

    return metrics

def compare_cka_across_conditions(pure_dir: str | Path, collapse_dir: str | Path, model_config: dict, sample_inputs: torch.Tensor) -> float:
    """
    Compare CKA of latest checkpoints from two conditions (e.g. pure vs collapse).
    """
    from src.model import ModularArithmeticTransformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_latest_ckpt_acts(ckpt_dir):
        checkpoints = sorted(
            [p for p in Path(ckpt_dir).glob("checkpoint_*.pt")],
            key=lambda p: int(p.stem.split('_')[1])
        )
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        model = ModularArithmeticTransformer(**model_config).to(device)
        try:
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state'])
        except Exception:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])

        model.eval()
        with torch.no_grad():
            # Get the pre-pooled representation
            tok = model.token_embed(sample_inputs.to(device))
            pos = model.pos_embed(torch.arange(2, device=device).unsqueeze(0).expand(sample_inputs.size(0), -1))
            h = tok + pos
            h = model.transformer(h)
            h = model.ln(h)
            acts = h.mean(dim=1)  # (batch, d_model)
        return acts

    acts_pure = get_latest_ckpt_acts(pure_dir)
    acts_collapse = get_latest_ckpt_acts(collapse_dir)

    if acts_pure is None or acts_collapse is None:
        return 0.0

    return compute_activation_cka(acts_pure, acts_collapse)
