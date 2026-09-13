import argparse
import copy
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.transplant.transplant_rescue import COMPONENT_PATTERNS, keys_for, load_run, build_model, make_loaders, evaluate_model
    from src.train import evaluate
except ImportError:
    from .transplant_rescue import COMPONENT_PATTERNS, keys_for, load_run, build_model, make_loaders, evaluate_model # type: ignore
    from ..train import evaluate # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_random_weights_like(param: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Generate random orthogonal/normal weights of the same shape."""
    w = param.detach().to(torch.float32)
    if w.ndim == 1:
        # For 1D (bias/norm), keep original mean/std but randomize
        return torch.randn(w.shape, generator=rng, device=w.device) * w.std() + w.mean()
    elif w.ndim == 2:
        # For 2D (linear), generate random orthogonal matrix to preserve spectral norm roughly
        # Or simply random normal with proper init scale. Here we match norm.
        random_matrix = torch.randn(w.shape, generator=rng, device=w.device)
        norm_ratio = w.norm() / (random_matrix.norm() + 1e-9)
        return random_matrix * norm_ratio
    else:
        # Fallback for higher dims
        random_tensor = torch.randn(w.shape, generator=rng, device=w.device)
        norm_ratio = w.norm() / (random_tensor.norm() + 1e-9)
        return random_tensor * norm_ratio


@contextmanager
def graft_context(
    host_model: nn.Module,
    donor_model: nn.Module,
    components: List[str],
    control_type: Literal["donor", "sham", "random"],
    rng: Optional[torch.Generator] = None
):
    """
    Context manager that temporarily grafts components into host_model.

    Args:
        host_model: The base model (collapsed). Modified in place.
        donor_model: The donor model (grokked).
        components: List of component names (e.g., 'token_embed', 'self_attn_in_proj').
        control_type:
            - 'donor': actual graft from donor.
            - 'sham': graft from host itself (should change nothing).
            - 'random': graft randomly initialized weights of same shape.
        rng: Random number generator for 'random' control.
    """
    # 1. Back up the original host state dict for these components
    host_sd = host_model.state_dict()
    donor_sd = donor_model.state_dict()

    backup_sd = {}
    patch_sd = {}

    for comp in components:
        # keys_for uses regex matching defined in COMPONENT_PATTERNS
        keys = keys_for(comp, host_sd)
        if not keys:
            logger.warning(f"No keys matched component {comp}")

        for k in keys:
            backup_sd[k] = host_sd[k].clone()

            if control_type == "donor":
                if k not in donor_sd:
                    raise ValueError(f"Key {k} missing in donor model.")
                patch_sd[k] = donor_sd[k].clone()
            elif control_type == "sham":
                patch_sd[k] = host_sd[k].clone()
            elif control_type == "random":
                if rng is None:
                    rng = torch.Generator(device=host_sd[k].device).manual_seed(0)
                patch_sd[k] = generate_random_weights_like(host_sd[k], rng)
            else:
                raise ValueError(f"Unknown control_type: {control_type}")

    # 2. Apply patch
    try:
        # strict=False because we only update part of the model
        host_model.load_state_dict(patch_sd, strict=False)
        yield
    finally:
        # 3. Restore original weights
        host_model.load_state_dict(backup_sd, strict=False)

def compute_recovery_fraction(graft_acc: float, host_acc: float, donor_acc: float) -> float:
    return (graft_acc - host_acc) / max(1e-9, donor_acc - host_acc)

def permutation_test(samples_a: np.ndarray, samples_b: np.ndarray, n_permutations: int = 10000) -> float:
    """Computes p-value for difference in means between two sets of samples using permutation test."""
    diff_obs = np.mean(samples_a) - np.mean(samples_b)
    combined = np.concatenate([samples_a, samples_b])
    n_a = len(samples_a)
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        diff_perm = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if diff_perm >= diff_obs:
            count += 1
    return count / n_permutations

def run_sweep(
    config_path: Path,
    host_dirs: List[Path],
    donor_dirs: List[Path],
    output_dir: Path,
    device: torch.device
):
    with config_path.open() as f:
        sweep_cfg = json.load(f)

    graft_sets = sweep_cfg.get("graft_sets", [])
    if not graft_sets:
        raise ValueError("No graft_sets found in config.")

    if len(host_dirs) != len(donor_dirs):
        raise ValueError("Number of host directories must match number of donor directories (paired seeds).")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seed_idx, (host_dir, donor_dir) in enumerate(zip(host_dirs, donor_dirs)):
        logger.info(f"Processing pair {seed_idx + 1}/{len(host_dirs)}")

        host_sd, host_cfg = load_run(host_dir)
        donor_sd, donor_cfg = load_run(donor_dir)

        host_model = build_model(host_cfg, device)
        host_model.load_state_dict(host_sd)

        donor_model = build_model(donor_cfg, device)
        donor_model.load_state_dict(donor_sd)

        # Ensure eval mode
        host_model.eval()
        donor_model.eval()

        train_loader, test_loader = make_loaders(host_cfg, device=device)

        # Baselines
        logger.info("Evaluating baselines...")
        host_metrics = evaluate_model(host_model, train_loader, test_loader, device)
        donor_metrics = evaluate_model(donor_model, train_loader, test_loader, device)

        host_acc = host_metrics["test_acc"]
        donor_acc = donor_metrics["test_acc"]

        results.append({
            "seed_idx": seed_idx,
            "graft_name": "baseline_host",
            "control_type": "none",
            "test_acc": host_acc,
            "test_loss": host_metrics["test_loss"],
            "recovery_fraction": 0.0
        })
        results.append({
            "seed_idx": seed_idx,
            "graft_name": "baseline_donor",
            "control_type": "none",
            "test_acc": donor_acc,
            "test_loss": donor_metrics["test_loss"],
            "recovery_fraction": 1.0
        })

        rng = torch.Generator(device=device).manual_seed(42 + seed_idx)

        for graft_set in graft_sets:
            name = graft_set["name"]
            components = graft_set["components"]

            for control in ["donor", "sham", "random"]:
                logger.info(f"Evaluating {name} ({control})...")
                with graft_context(host_model, donor_model, components, control, rng):
                    metrics = evaluate_model(host_model, train_loader, test_loader, device)
                    graft_acc = metrics["test_acc"]
                    recovery = compute_recovery_fraction(graft_acc, host_acc, donor_acc)

                    results.append({
                        "seed_idx": seed_idx,
                        "graft_name": name,
                        "control_type": control,
                        "test_acc": graft_acc,
                        "test_loss": metrics["test_loss"],
                        "recovery_fraction": recovery
                    })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "sweep_results.csv", index=False)

    # Aggregation
    summary = []

    unique_grafts = df[~df["graft_name"].str.startswith("baseline_")]["graft_name"].unique()

    for graft in unique_grafts:
        donor_vals = df[(df["graft_name"] == graft) & (df["control_type"] == "donor")]["test_acc"].values
        sham_vals = df[(df["graft_name"] == graft) & (df["control_type"] == "sham")]["test_acc"].values
        rand_vals = df[(df["graft_name"] == graft) & (df["control_type"] == "random")]["test_acc"].values

        donor_recov = df[(df["graft_name"] == graft) & (df["control_type"] == "donor")]["recovery_fraction"].values

        p_val = permutation_test(donor_vals, sham_vals)

        summary.append({
            "graft_set": graft,
            "donor_acc_mean": np.mean(donor_vals),
            "donor_acc_std": np.std(donor_vals),
            "sham_acc_mean": np.mean(sham_vals),
            "sham_acc_std": np.std(sham_vals),
            "rand_acc_mean": np.mean(rand_vals),
            "rand_acc_std": np.std(rand_vals),
            "recovery_fraction_mean": np.mean(donor_recov),
            "recovery_fraction_std": np.std(donor_recov),
            "p_value_vs_sham": p_val,
            "significant": p_val < 0.05
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)

    with open(output_dir / "sweep_summary.md", "w") as f:
        f.write("# Circuit Transplant Sweep Summary\n\n")
        f.write(summary_df.to_markdown(index=False))

    logger.info(f"Sweep complete. Results written to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run generalized transplant sweep.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config defining graft_sets.")
    parser.add_argument("--host-dirs", type=Path, nargs='+', required=True, help="List of host (collapsed) run directories.")
    parser.add_argument("--donor-dirs", type=Path, nargs='+', required=True, help="List of donor (grokked) run directories, matching seeds of host-dirs.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/transplant_sweep"), help="Output directory.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_sweep(args.config, args.host_dirs, args.donor_dirs, args.output_dir, device)

if __name__ == "__main__":
    main()
