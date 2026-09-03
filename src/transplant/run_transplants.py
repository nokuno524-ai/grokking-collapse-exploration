import argparse
import copy
import json
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.train import compute_fourier_concentration, evaluate
    from src.transplant.circuits import swap_attention_head, swap_mlp, swap_layernorm
except ImportError:
    from model import ModularArithmeticTransformer  # type: ignore
    from data import DatasetConfig, generate_modular_arithmetic  # type: ignore
    from train import compute_fourier_concentration, evaluate  # type: ignore
    from transplant.circuits import swap_attention_head, swap_mlp, swap_layernorm # type: ignore


# Fix random basis swap logic
def random_basis_swap(weight: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Return a tensor with the same shape and spectrum as `weight` but a
    random orthonormal basis.
    """
    w = weight.detach().to(torch.float32).clone()
    if w.ndim == 1:
        idx = torch.randperm(w.numel(), generator=rng)
        return w[idx]
    if w.ndim != 2:
        orig_shape = w.shape
        w2 = w.reshape(w.shape[0], -1)
        out = random_basis_swap(w2, rng)
        return out.reshape(orig_shape)

    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng))
    Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng))

    return (Ur * S) @ Vr.T

def load_run(run_dir: Path, step: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Return (state_dict, config) for the given run."""
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


def build_model(cfg: dict, device: torch.device) -> ModularArithmeticTransformer:
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)


def make_loaders(
    cfg: dict, batch_size: int = 512, device: torch.device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader]:
    dc = DatasetConfig(
        prime=int(cfg.get("prime", 59)),
        train_fraction=float(cfg.get("train_fraction", 0.3)),
        collapse_level=float(cfg.get("collapse_level", 0.0)),
        collapse_severity=float(cfg.get("collapse_severity", 0.5)),
        noise_fraction=float(cfg.get("noise_fraction", 0.0)),
        seed=int(cfg.get("seed", 42)),
    )
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(dc)
    train_ds = TensorDataset(train_in, train_tgt)
    test_ds = TensorDataset(test_in, test_tgt)
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def check_input_independent_constant_attention(model: nn.Module, loader: DataLoader, device: torch.device) -> List[Tuple[int, int]]:
    """Identifies heads that perform input-independent constant attention.

    Returns a list of (layer_idx, head_idx) that are constant.
    """
    model.eval()
    constant_heads = []

    layer_weights = {}

    # We will use hooks to intercept the input and output of the MultiheadAttention blocks,
    # but PyTorch's MultiheadAttention doesn't return weights by default.
    # We must patch the forward pass of MHA, but doing it correctly by hooking into the model.
    # A cleaner approach: patch the forward pass temporarily to set need_weights=True and save the weights.

    original_forwards = {}

    def make_patched_forward(l_idx, layer):
        orig_forward = layer.self_attn.forward
        original_forwards[l_idx] = orig_forward

        def _forward(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                out, weights = orig_forward(*args, **kwargs)
            layer_weights[l_idx] = weights.detach()
            return out, weights # Layer expects out, weights format if we patched correctly? No, standard PyTorch nn.TransformerEncoderLayer expects out to be just the tensor if need_weights=False but we are inside the TransformerEncoderLayer so we don't control its expectations easily.
            # Actually, standard nn.TransformerEncoderLayer forward:
            # x = self.self_attn(x, x, x, ...)[0]
            # So returning a tuple is safe for the inner self_attn call as the parent layer extracts [0].
        return _forward

    with torch.no_grad():
        try:
            x, _ = next(iter(loader))
            x = x.to(device)

            # Apply patches
            for l_idx, layer in enumerate(model.transformer.layers):
                layer.self_attn.forward = make_patched_forward(l_idx, layer)

            # Forward pass through the entire model to populate layer_weights
            _ = model(x)

            for l_idx in range(len(model.transformer.layers)):
                if l_idx in layer_weights:
                    weights = layer_weights[l_idx]
                    # weights shape: (batch, n_heads, seq_len, seq_len)
                    var_across_batch = weights.var(dim=0).mean(dim=(1, 2)) # (n_heads,)
                    for h_idx in range(model.n_heads):
                        if var_across_batch[h_idx].item() < 1e-4:
                            constant_heads.append((l_idx, h_idx))

        finally:
            # Always restore originals
            for l_idx, layer in enumerate(model.transformer.layers):
                if l_idx in original_forwards:
                    layer.self_attn.forward = original_forwards[l_idx]

    return constant_heads


@dataclass
class TransplantResult:
    donor_condition: str
    recipient_condition: str
    layer_idx: int
    head_idx: Optional[int]
    component_type: str  # 'head', 'mlp', 'ln1', 'ln2', 'all'
    baseline_acc: float
    transplant_acc: float
    acc_delta: float
    is_constant_attention: bool = False


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    _, acc = evaluate(model, loader, device)
    return acc

def run_transplants_for_pair(
    donor_dir: Path,
    recipient_dir: Path,
    donor_condition: str,
    recipient_condition: str,
    device: torch.device
) -> List[TransplantResult]:
    donor_sd, donor_cfg = load_run(donor_dir)
    recip_sd, recip_cfg = load_run(recipient_dir)

    if int(donor_cfg.get("seed", -1)) != int(recip_cfg.get("seed", -2)):
        raise ValueError("Donor and recipient seeds must match to avoid train/test split leakage.")

    _, test_loader = make_loaders(recip_cfg, device=device)

    model = build_model(recip_cfg, device)

    # Baseline recipient
    model.load_state_dict(recip_sd, strict=True)
    baseline_acc = evaluate_model(model, test_loader, device)

    # Baseline donor (sanity check)
    model.load_state_dict(donor_sd, strict=True)
    donor_acc = evaluate_model(model, test_loader, device)

    # Get constant heads for the donor
    train_loader, _ = make_loaders(recip_cfg, device=device)
    constant_heads = check_input_independent_constant_attention(model, train_loader, device)

    results = []

    n_layers = int(recip_cfg.get("n_layers", 1))
    n_heads = int(recip_cfg.get("n_heads", 4))
    d_model = int(recip_cfg.get("d_model", 128))

    for l_idx in range(n_layers):
        # 1. Heads
        for h_idx in range(n_heads):
            patched_sd = swap_attention_head(recip_sd, donor_sd, l_idx, h_idx, n_heads, d_model)
            model.load_state_dict(patched_sd, strict=True)
            acc = evaluate_model(model, test_loader, device)
            is_const = (l_idx, h_idx) in constant_heads
            results.append(TransplantResult(
                donor_condition, recipient_condition, l_idx, h_idx, "head",
                baseline_acc, acc, acc - baseline_acc, is_const
            ))

        # 2. MLP
        patched_sd = swap_mlp(recip_sd, donor_sd, l_idx)
        model.load_state_dict(patched_sd, strict=True)
        acc = evaluate_model(model, test_loader, device)
        results.append(TransplantResult(
            donor_condition, recipient_condition, l_idx, None, "mlp",
            baseline_acc, acc, acc - baseline_acc
        ))

        # 3. LayerNorms
        patched_sd = swap_layernorm(recip_sd, donor_sd, l_idx, 1)
        model.load_state_dict(patched_sd, strict=True)
        acc = evaluate_model(model, test_loader, device)
        results.append(TransplantResult(
            donor_condition, recipient_condition, l_idx, None, "ln1",
            baseline_acc, acc, acc - baseline_acc
        ))

        patched_sd = swap_layernorm(recip_sd, donor_sd, l_idx, 2)
        model.load_state_dict(patched_sd, strict=True)
        acc = evaluate_model(model, test_loader, device)
        results.append(TransplantResult(
            donor_condition, recipient_condition, l_idx, None, "ln2",
            baseline_acc, acc, acc - baseline_acc
        ))

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure-dirs", type=str, nargs="+", required=True)
    ap.add_argument("--low-dirs", type=str, nargs="+", required=True)
    ap.add_argument("--severe-dirs", type=str, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/transplant"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We assume dirs are sorted by seed so that zip matches seeds
    pure_dirs = [Path(p) for p in args.pure_dirs]
    low_dirs = [Path(p) for p in args.low_dirs]
    severe_dirs = [Path(p) for p in args.severe_dirs]

    all_results = []

    # donor -> recipient matrices
    scenarios = [
        ("pure", "low", pure_dirs, low_dirs),
        ("pure", "severe", pure_dirs, severe_dirs),
        ("low", "severe", low_dirs, severe_dirs),
    ]

    for donor_cond, recip_cond, donors, recips in scenarios:
        print(f"Running transplants: {donor_cond} -> {recip_cond}")
        for donor_dir, recip_dir in zip(donors, recips):
            try:
                res = run_transplants_for_pair(donor_dir, recip_dir, donor_cond, recip_cond, device)
                all_results.extend(res)
            except Exception as e:
                print(f"Skipping pair {donor_dir} -> {recip_dir} due to error: {e}")

    # Save raw results
    df = pd.DataFrame([asdict(r) for r in all_results])
    df.to_csv(args.output_dir / "transplant_raw.csv", index=False)
    print(f"Saved raw results to {args.output_dir / 'transplant_raw.csv'}")

if __name__ == "__main__":
    main()
