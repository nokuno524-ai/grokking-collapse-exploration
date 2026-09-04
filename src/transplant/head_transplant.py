import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.transplant.transplant_rescue import load_run, make_loaders, evaluate_model, random_basis_swap
except ImportError:
    from transplant_rescue import load_run, make_loaders, evaluate_model, random_basis_swap  # type: ignore

try:
    from src.model import ModularArithmeticTransformer
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from model import ModularArithmeticTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_head_indices(d_model: int, n_heads: int, head_idx: int) -> slice:
    """Return the slice indices for a given head in the hidden dimension."""
    d_head = d_model // n_heads
    return slice(head_idx * d_head, (head_idx + 1) * d_head)


def patch_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int,
    mode: str = "swap",
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Patch a specific attention head in the state dictionary.

    Args:
        base_sd: The original model state dict.
        donor_sd: State dict to copy the head from (if mode == 'swap').
        layer_idx: Index of the transformer layer.
        head_idx: Index of the attention head to patch.
        d_model: Model hidden dimension size.
        n_heads: Total number of attention heads.
        mode: One of 'swap', 'zero', 'random'.
        rng: Random number generator for 'random' mode.

    Returns:
        A new state dict with the head patched.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    head_slice = get_head_indices(d_model, n_heads, head_idx)

    in_proj_w_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_b_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
    out_proj_w_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"
    out_proj_b_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.bias"

    if mode == "swap":
        if donor_sd is None:
            raise ValueError("donor_sd must be provided for 'swap' mode")

        # in_proj_weight is shape (3 * d_model, d_model) -> Q, K, V
        for i in range(3):
            qkv_offset = i * d_model
            idx_start = qkv_offset + head_slice.start
            idx_end = qkv_offset + head_slice.stop
            out_sd[in_proj_w_key][idx_start:idx_end, :] = donor_sd[in_proj_w_key][idx_start:idx_end, :]

        if in_proj_b_key in out_sd:
            for i in range(3):
                qkv_offset = i * d_model
                idx_start = qkv_offset + head_slice.start
                idx_end = qkv_offset + head_slice.stop
                out_sd[in_proj_b_key][idx_start:idx_end] = donor_sd[in_proj_b_key][idx_start:idx_end]

        # out_proj.weight is shape (d_model, d_model), heads are on the second dimension
        out_sd[out_proj_w_key][:, head_slice] = donor_sd[out_proj_w_key][:, head_slice]

        # Note: out_proj.bias is per-layer, not per-head, so we do not patch it or we'd overwrite other heads.

    elif mode == "zero":
        for i in range(3):
            qkv_offset = i * d_model
            idx_start = qkv_offset + head_slice.start
            idx_end = qkv_offset + head_slice.stop
            out_sd[in_proj_w_key][idx_start:idx_end, :].zero_()
            if in_proj_b_key in out_sd:
                out_sd[in_proj_b_key][idx_start:idx_end].zero_()

        out_sd[out_proj_w_key][:, head_slice].zero_()

    elif mode == "random":
        if rng is None:
            rng = torch.Generator().manual_seed(42)

        # Randomize Q, K, V
        for i in range(3):
            qkv_offset = i * d_model
            idx_start = qkv_offset + head_slice.start
            idx_end = qkv_offset + head_slice.stop
            # Shape is (d_head, d_model)
            slice_w = out_sd[in_proj_w_key][idx_start:idx_end, :]
            # We want to randomize it while keeping the shape and spectrum.
            # random_basis_swap expects (..., in_features) ?? Actually let's look at its implementation.
            # In transplant_rescue it does:
            # U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            # Vr = ... random orthonormal
            # return (U * S) @ Vr.T
            # Our slice_w is (d_head, d_model). It's not square.
            # random_basis_swap handles non-square by randomizing V.
            out_sd[in_proj_w_key][idx_start:idx_end, :] = random_basis_swap(slice_w, rng)

        # Randomize O
        slice_out_w = out_sd[out_proj_w_key][:, head_slice] # (d_model, d_head)
        out_sd[out_proj_w_key][:, head_slice] = random_basis_swap(slice_out_w, rng)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out_sd



def process_single_pair(
    pure_run: Path,
    contam_run: Path,
    pure_step: Optional[int] = None,
    contam_step: Optional[int] = None,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, Any], List[Tuple[Tuple[int, int], float]], np.ndarray, float, float]:
    pure_sd, pure_cfg = load_run(pure_run, pure_step)
    contam_sd, contam_cfg = load_run(contam_run, contam_step)

    # Assert matched configs
    for fld in ("prime", "d_model", "n_heads", "d_ff", "n_layers"):
        if pure_cfg.get(fld) != contam_cfg.get(fld):
            raise ValueError(f"architecture mismatch on {fld}")

    d_model = int(pure_cfg.get("d_model", 128))
    n_heads = int(pure_cfg.get("n_heads", 4))
    n_layers = int(pure_cfg.get("n_layers", 1))

    pure_model = ModularArithmeticTransformer(**pure_cfg)
    pure_model.load_state_dict(pure_sd, strict=True)
    pure_model.to(device).eval()

    contam_model = ModularArithmeticTransformer(**contam_cfg)
    contam_model.load_state_dict(contam_sd, strict=True)
    contam_model.to(device).eval()

    train_loader, test_loader = make_loaders(contam_cfg, pure_cfg)

    base_eval = evaluate_model(contam_model, train_loader, test_loader, device)
    contam_test_acc = base_eval["test_acc"]
    pure_eval = evaluate_model(pure_model, train_loader, test_loader, device)
    pure_test_acc = pure_eval["test_acc"]

    rng = torch.Generator().manual_seed(seed)
    results = {}
    matrix = np.zeros((n_layers, n_heads))

    # Single-head transplants: Pure -> Contam (Rescue)
    for l in range(n_layers):
        for h in range(n_heads):
            # Swap
            swapped_sd = patch_head(contam_sd, pure_sd, l, h, d_model, n_heads, mode="swap")
            contam_model.load_state_dict(swapped_sd, strict=True)
            swap_eval = evaluate_model(contam_model, train_loader, test_loader, device)

            # Zero
            zero_sd = patch_head(contam_sd, None, l, h, d_model, n_heads, mode="zero")
            contam_model.load_state_dict(zero_sd, strict=True)
            zero_eval = evaluate_model(contam_model, train_loader, test_loader, device)

            # Random
            rand_sd = patch_head(contam_sd, None, l, h, d_model, n_heads, mode="random", rng=rng)
            contam_model.load_state_dict(rand_sd, strict=True)
            rand_eval = evaluate_model(contam_model, train_loader, test_loader, device)

            results[f"L{l}H{h}"] = {
                "swap_test_acc": swap_eval["test_acc"],
                "zero_test_acc": zero_eval["test_acc"],
                "rand_test_acc": rand_eval["test_acc"],
            }
            matrix[l, h] = swap_eval["test_acc"] - contam_test_acc

    # Greedy search for minimal rescue set
    current_sd = {k: v.clone() for k, v in contam_sd.items()}
    remaining_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    greedy_order = []

    while remaining_heads:
        best_acc = -1.0
        best_head = None
        best_sd = None

        for l, h in remaining_heads:
            test_sd = patch_head(current_sd, pure_sd, l, h, d_model, n_heads, mode="swap")
            contam_model.load_state_dict(test_sd, strict=True)
            ev = evaluate_model(contam_model, train_loader, test_loader, device)
            if ev["test_acc"] > best_acc:
                best_acc = ev["test_acc"]
                best_head = (l, h)
                best_sd = test_sd

        greedy_order.append((best_head, best_acc))
        remaining_heads.remove(best_head)
        current_sd = best_sd

    return results, greedy_order, matrix, pure_test_acc, contam_test_acc

def run_head_transplants(
    pure_runs: List[Path],
    contam_runs: List[Path],
    output_dir: Path,
    pure_step: Optional[int] = None,
    contam_step: Optional[int] = None,
    seed: int = 42,
) -> None:
    """
    Run head-level transplants across all layers and heads to attribute grokking to specific heads.
    Outputs a JSON file, a heatmap of test accuracy deltas, and a markdown report.
    Supports multiple seeds for confidence intervals.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if len(pure_runs) != len(contam_runs):
        raise ValueError(f"Number of pure runs ({len(pure_runs)}) must match contam runs ({len(contam_runs)})")

    all_results = []
    all_matrices = []
    all_pure_accs = []
    all_contam_accs = []

    for i, (p_run, c_run) in enumerate(zip(pure_runs, contam_runs)):
        logging.info(f"Processing pair {i+1}/{len(pure_runs)}: {p_run.name} vs {c_run.name}")
        try:
            res, greedy, mat, p_acc, c_acc = process_single_pair(
                p_run, c_run, pure_step, contam_step, seed, device
            )
            all_results.append((res, greedy))
            all_matrices.append(mat)
            all_pure_accs.append(p_acc)
            all_contam_accs.append(c_acc)
        except Exception as e:
            logging.error(f"Error processing pair {p_run} vs {c_run}: {e}")
            raise e

    output_dir.mkdir(parents=True, exist_ok=True)

    mean_matrix = np.mean(all_matrices, axis=0) if all_matrices else np.array([])

    n_layers, n_heads = mean_matrix.shape if all_matrices else (0, 0)

    # Calculate CIs (95% roughly 1.96 * std / sqrt(n))
    if len(all_matrices) > 1:
        std_matrix = np.std(all_matrices, axis=0)
        ci_matrix = 1.96 * std_matrix / np.sqrt(len(all_matrices))
    else:
        ci_matrix = np.zeros_like(mean_matrix)

    # Save JSON
    with open(output_dir / "head_transplant.json", "w") as f:
        json.dump({
            "pure_runs": [str(p) for p in pure_runs],
            "contam_runs": [str(c) for c in contam_runs],
            "mean_pure_test_acc": float(np.mean(all_pure_accs)) if all_pure_accs else 0.0,
            "mean_contam_test_acc": float(np.mean(all_contam_accs)) if all_contam_accs else 0.0,
            "runs": [
                {
                    "pure_run": str(p),
                    "contam_run": str(c),
                    "pure_test_acc": float(p_acc),
                    "contam_test_acc": float(c_acc),
                    "results": res,
                    "greedy_search": [{"head": f"L{l}H{h}", "test_acc": float(acc)} for (l, h), acc in greedy]
                }
                for p, c, p_acc, c_acc, (res, greedy) in zip(pure_runs, contam_runs, all_pure_accs, all_contam_accs, all_results)
            ]
        }, f, indent=2)

    # Plot Heatmap
    if n_layers > 0 and n_heads > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(mean_matrix, cmap="RdBu", vmin=-1.0, vmax=1.0)
        fig.colorbar(cax)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{i}" for i in range(n_heads)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title("Single-Head Transplant Rescue (Mean Test Acc Delta)")
        fig.savefig(output_dir / "head_rescue_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Markdown Report
    with open(output_dir / "head_transplant.md", "w") as f:
        f.write("# Head-Level Transplant Analysis\n\n")
        f.write(f"Processed {len(pure_runs)} pairs of runs.\n\n")
        m_pure = np.mean(all_pure_accs) if all_pure_accs else 0.0
        ci_pure = 1.96*np.std(all_pure_accs)/np.sqrt(len(all_pure_accs)) if len(all_pure_accs)>1 else 0.0
        m_contam = np.mean(all_contam_accs) if all_contam_accs else 0.0
        ci_contam = 1.96*np.std(all_contam_accs)/np.sqrt(len(all_contam_accs)) if len(all_contam_accs)>1 else 0.0

        f.write(f"- Mean Baseline Pure: {m_pure:.3f} +- {ci_pure:.3f}\n")
        f.write(f"- Mean Baseline Contam: {m_contam:.3f} +- {ci_contam:.3f}\n\n")

        f.write("## Single Head Rescue (Pure -> Contam)\n")
        f.write("| Head | Mean Swap | Mean Zero | Mean Random | Mean Delta from Contam | 95% CI of Delta |\n")
        f.write("|---|---|---|---|---|---|\n")

        for l in range(n_layers):
            for h in range(n_heads):
                h_key = f"L{l}H{h}"
                swaps = [res[h_key]["swap_test_acc"] for res, _ in all_results]
                zeros = [res[h_key]["zero_test_acc"] for res, _ in all_results]
                rands = [res[h_key]["rand_test_acc"] for res, _ in all_results]

                m_swap = np.mean(swaps)
                m_zero = np.mean(zeros)
                m_rand = np.mean(rands)
                m_delta = mean_matrix[l, h]
                ci_delta = ci_matrix[l, h]

                f.write(f"| {h_key} | {m_swap:.3f} | {m_zero:.3f} | {m_rand:.3f} | {m_delta:+.3f} | +-{ci_delta:.3f} |\n")

        f.write("\n## Example Greedy Rescue Set (from run 1)\n")
        if all_results:
            _, first_greedy = all_results[0]
            f.write("Iteratively adding heads to contam from pure:\n")
            for i, ((l, h), acc) in enumerate(first_greedy):
                f.write(f"{i+1}. Add L{l}H{h} -> Test Acc: {acc:.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-runs", type=Path, nargs="+", required=True, help="List of pure run directories")
    parser.add_argument("--contam-runs", type=Path, nargs="+", required=True, help="List of contaminated run directories")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/head_transplant"))
    parser.add_argument("--pure-step", type=int, default=None)
    parser.add_argument("--contam-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # If run as module without args just show help and exit
    args = parser.parse_args()
    run_head_transplants(
        args.pure_runs, args.contam_runs, args.output_dir,
        args.pure_step, args.contam_step, args.seed
    )
