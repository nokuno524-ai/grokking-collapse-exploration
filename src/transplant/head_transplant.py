"""
Fine-grained circuit transplantation tools.
Supports layer-wise, head-wise, bidirectional, and hybrid transplants
between pure-trained and collapse-trained checkpoints.
"""

import argparse
import torch
import json
from typing import Dict, Tuple
from pathlib import Path
from src.model import ModularArithmeticTransformer
from src.checkpoint import load_run
from src.transplant.transplant_rescue import evaluate_model, make_loaders, COMPONENT_PATTERNS

def split_in_proj_weight(in_proj_weight: torch.Tensor, d_model: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split PyTorch's fused in_proj_weight into Q, K, V chunks."""
    assert in_proj_weight.shape[0] == 3 * d_model, f"Expected {3 * d_model}, got {in_proj_weight.shape[0]}"
    q, k, v = in_proj_weight.chunk(3, dim=0)
    return q, k, v

def ablate_head(
    in_proj_weight: torch.Tensor,
    d_model: int,
    n_heads: int,
    head_idx: int,
    ablation_type: str = "zero"
) -> torch.Tensor:
    """Ablates a specific head from the fused in_proj_weight matrix."""
    d_head = d_model // n_heads
    q, k, v = split_in_proj_weight(in_proj_weight.clone(), d_model)

    start = head_idx * d_head
    end = (head_idx + 1) * d_head

    for chunk in [q, k, v]:
        if ablation_type == "zero":
            chunk[start:end, :] = 0.0
        elif ablation_type == "mean":
            chunk[start:end, :] = chunk[start:end, :].mean(dim=0, keepdim=True).expand_as(chunk[start:end, :])
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

    return torch.cat([q, k, v], dim=0)

def transplant_head(
    target_in_proj: torch.Tensor,
    donor_in_proj: torch.Tensor,
    d_model: int,
    n_heads: int,
    head_idx: int
) -> torch.Tensor:
    """Transplants a specific head from donor to target fused in_proj_weight matrix."""
    d_head = d_model // n_heads
    target_q, target_k, target_v = split_in_proj_weight(target_in_proj.clone(), d_model)
    donor_q, donor_k, donor_v = split_in_proj_weight(donor_in_proj, d_model)

    start = head_idx * d_head
    end = (head_idx + 1) * d_head

    target_q[start:end, :] = donor_q[start:end, :]
    target_k[start:end, :] = donor_k[start:end, :]
    target_v[start:end, :] = donor_v[start:end, :]

    return torch.cat([target_q, target_k, target_v], dim=0)

def zero_ablate_component(state_dict: Dict[str, torch.Tensor], component_key: str) -> Dict[str, torch.Tensor]:
    """Zero ablates an entire component in the state dict."""
    new_sd = {k: v.clone() for k, v in state_dict.items()}
    if component_key in new_sd:
        new_sd[component_key] = torch.zeros_like(new_sd[component_key])
    return new_sd

def mean_ablate_component(state_dict: Dict[str, torch.Tensor], component_key: str) -> Dict[str, torch.Tensor]:
    """Mean ablates an entire component in the state dict."""
    new_sd = {k: v.clone() for k, v in state_dict.items()}
    if component_key in new_sd:
        new_sd[component_key] = new_sd[component_key].mean(dim=0, keepdim=True).expand_as(new_sd[component_key])
    return new_sd

def main():
    parser = argparse.ArgumentParser(description="Head-level circuit transplant tools")
    parser.add_argument("--pure-runs", nargs="+", type=Path, help="Paths to pure runs")
    parser.add_argument("--contam-runs", nargs="+", type=Path, help="Paths to contaminated runs")
    parser.add_argument("--ablate-type", type=str, choices=["zero", "mean"], default="zero")
    parser.add_argument("--head-idx", type=int, default=None, help="Specific head index to ablate/transplant")
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/ablation_ladder"), help="Output dir")
    args = parser.parse_args()

    if not args.pure_runs and not args.contam_runs:
        print("No runs provided.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    # Process pairs if both provided
    if args.pure_runs and args.contam_runs:
        for pure_run, contam_run in zip(args.pure_runs, args.contam_runs):
            try:
                pure_sd, pure_cfg = load_run(pure_run)
                contam_sd, contam_cfg = load_run(contam_run)

                # Check config match
                if pure_cfg.get("seed") != contam_cfg.get("seed"):
                    print(f"Warning: Seeds mismatch for {pure_run} and {contam_run}")

                model = ModularArithmeticTransformer(
                    prime=pure_cfg.get("prime", 59),
                    d_model=pure_cfg.get("d_model", 128),
                    n_heads=pure_cfg.get("n_heads", 4),
                    d_ff=pure_cfg.get("d_ff", 512),
                    n_layers=pure_cfg.get("n_layers", 1),
                ).to(device)

                train_loader, test_loader = make_loaders(pure_cfg, device=device)

                # 1. Baseline Pure
                model.load_state_dict(pure_sd, strict=False)
                baseline_pure = evaluate_model(model, train_loader, test_loader, device)

                # 2. Baseline Contam
                model.load_state_dict(contam_sd, strict=False)
                baseline_contam = evaluate_model(model, train_loader, test_loader, device)

                # 3. Transplant Head if specified
                if args.head_idx is not None:
                    # contam with pure head
                    patched_sd = {k: v.clone() for k, v in contam_sd.items()}
                    proj_key = f"transformer.layers.{args.layer_idx}.self_attn.in_proj_weight"

                    if proj_key in patched_sd and proj_key in pure_sd:
                        patched_sd[proj_key] = transplant_head(
                            patched_sd[proj_key], pure_sd[proj_key],
                            pure_cfg.get("d_model", 128), pure_cfg.get("n_heads", 4), args.head_idx
                        )
                        model.load_state_dict(patched_sd, strict=False)
                        transplant_res = evaluate_model(model, train_loader, test_loader, device)

                        results.append({
                            "run_pair": f"{pure_run.name}_{contam_run.name}",
                            "baseline_pure_acc": baseline_pure["test_acc"],
                            "baseline_contam_acc": baseline_contam["test_acc"],
                            f"transplant_L{args.layer_idx}_H{args.head_idx}_acc": transplant_res["test_acc"]
                        })
            except Exception as e:
                print(f"Error processing {pure_run}: {e}")

    # Process individual runs for ablation
    if args.pure_runs:
        for run in args.pure_runs:
            try:
                sd, cfg = load_run(run)
                model = ModularArithmeticTransformer(
                    prime=cfg.get("prime", 59),
                    d_model=cfg.get("d_model", 128),
                    n_heads=cfg.get("n_heads", 4),
                    d_ff=cfg.get("d_ff", 512),
                    n_layers=cfg.get("n_layers", 1),
                ).to(device)
                train_loader, test_loader = make_loaders(cfg, device=device)

                model.load_state_dict(sd, strict=False)
                base = evaluate_model(model, train_loader, test_loader, device)

                if args.head_idx is not None:
                    proj_key = f"transformer.layers.{args.layer_idx}.self_attn.in_proj_weight"
                    if proj_key in sd:
                        patched_sd = {k: v.clone() for k, v in sd.items()}
                        patched_sd[proj_key] = ablate_head(
                            patched_sd[proj_key], cfg.get("d_model", 128), cfg.get("n_heads", 4), args.head_idx, args.ablate_type
                        )
                        model.load_state_dict(patched_sd, strict=False)
                        ablate = evaluate_model(model, train_loader, test_loader, device)

                        results.append({
                            "run": str(run),
                            "baseline_acc": base["test_acc"],
                            f"ablate_{args.ablate_type}_L{args.layer_idx}_H{args.head_idx}_acc": ablate["test_acc"]
                        })
            except Exception as e:
                print(f"Error processing {run}: {e}")

    with open(args.output_dir / "head_transplant_results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Saved results to {args.output_dir / 'head_transplant_results.json'}")

if __name__ == "__main__":
    main()
