import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
import scipy.sparse.linalg

def get_effective_rank(W: torch.Tensor) -> float:
    """Compute effective rank of the weight matrix."""
    # SVD
    s = torch.linalg.svdvals(W)
    s = s / s.sum()
    entropy = -(s * torch.log(s + 1e-10)).sum()
    return torch.exp(entropy).item()

def get_pca_spectrum(W: torch.Tensor) -> List[float]:
    """Compute the singular values (PCA spectrum) of the matrix."""
    s = torch.linalg.svdvals(W)
    return s.tolist()

def get_pairwise_cosine_similarity(W: torch.Tensor) -> np.ndarray:
    """
    Compute pairwise cosine similarities for all tokens.
    For (a+b)%p, all tokens are digits.
    """
    # Normalize rows
    W_norm = W / W.norm(dim=1, keepdim=True)
    # Cosine similarity matrix
    sim = W_norm @ W_norm.T

    # We want the histogram of the upper triangle (excluding diagonal)
    n = W.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)
    return sim[triu_indices[0], triu_indices[1]].detach().cpu().numpy()

def get_hidden_states_pca(hidden_states: torch.Tensor) -> np.ndarray:
    """
    Project hidden states onto 2D PCA.
    hidden_states: (batch_size, d_model)
    Returns: (batch_size, 2)
    """
    if hidden_states.shape[0] < 2:
        # Fallback if too few samples
        return np.zeros((hidden_states.shape[0], 2))

    pca = PCA(n_components=2)
    h_np = hidden_states.detach().cpu().numpy()
    try:
        proj = pca.fit_transform(h_np)
        return proj
    except Exception:
        return np.zeros((hidden_states.shape[0], 2))

import json
import re
from pathlib import Path
from typing import Iterator

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from torch.utils.data import DataLoader, TensorDataset

def get_eval_batch(run_dir: Path, batch_size: int = 512) -> torch.Tensor:
    """Load the fixed eval batch (train_inputs) for a given run."""
    with open(run_dir / "results.json") as f:
        config_dict = json.load(f)["config"]
    config = DatasetConfig(**config_dict)
    train_in, _, _, _ = generate_modular_arithmetic(config)
    return train_in[:batch_size]

def extract_hidden_states(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run forward pass and extract hidden states before output head."""
    model.eval()
    with torch.no_grad():
        tok = model.token_embed(inputs)
        positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos
        h = model.transformer(h)
        h = model.ln(h)
        h = h.mean(dim=1)
    return h

def process_checkpoint(ckpt_path: Path, eval_batch: torch.Tensor, device: torch.device) -> dict:
    """Process a single checkpoint and return geometry metrics."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    step = ckpt["step"]

    # Init model
    config = DatasetConfig(**ckpt.get("config", {}))
    model = ModularArithmeticTransformer(prime=config.prime).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Token embed geometry
    W_emb = model.token_embed.weight
    eff_rank = get_effective_rank(W_emb)
    pca_spec = get_pca_spectrum(W_emb)
    cos_sim_hist = get_pairwise_cosine_similarity(W_emb).tolist()

    # Hidden states geometry
    h = extract_hidden_states(model, eval_batch.to(device))
    h_pca = get_hidden_states_pca(h).tolist()

    return {
        "step": step,
        "effective_rank": eff_rank,
        "pca_spectrum": pca_spec,
        "cosine_similarity_histogram": cos_sim_hist,
        "hidden_states_pca": h_pca
    }

def run_tracker(run_dir: Path, device: torch.device) -> None:
    """Iterate over checkpoints in a run directory and append to JSONL."""
    run_dir = Path(run_dir)
    if not (run_dir / "results.json").exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")

    eval_batch = get_eval_batch(run_dir)
    out_file = run_dir / "embedding_geometry.jsonl"

    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

    with open(out_file, "w") as f:
        for ckpt_path in ckpts:
            metrics = process_checkpoint(ckpt_path, eval_batch, device)
            f.write(json.dumps(metrics) + "\n")

def compare_trajectories(pure_dir: Path, contam_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Loader function to read JSONL files for pure vs collapsed runs."""
    def load_jsonl(p: Path) -> List[dict]:
        with open(p) as f:
            return [json.loads(line) for line in f]

    pure_file = pure_dir / "embedding_geometry.jsonl"
    contam_file = contam_dir / "embedding_geometry.jsonl"

    if not pure_file.exists() or not contam_file.exists():
        raise FileNotFoundError("Missing embedding_geometry.jsonl in one or both directories. Run run_tracker first.")

    return load_jsonl(pure_file), load_jsonl(contam_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tracker(Path(args.run_dir), device)
