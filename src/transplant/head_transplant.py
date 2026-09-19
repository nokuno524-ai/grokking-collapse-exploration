import argparse
import json
import re
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import ModularArithmeticTransformer
from src.transplant.patching import patch_attention_head, patch_module, verify_patch
from src.checkpoint import load_run
from src.data import DatasetConfig, generate_modular_arithmetic
from src.transplant_rescue import evaluate, compute_fourier_concentration, evaluate_model, rescue_train
from tqdm import tqdm

def make_loaders(cfg: dict, batch_size: int = 512, device: torch.device = torch.device("cpu")) -> Tuple[DataLoader, DataLoader]:
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

def get_component_list(d_model: int, n_heads: int, n_layers: int = 1) -> List[Dict[str, Any]]:
    """Returns a list of components to patch."""
    components = [
        {"type": "module", "name": "token_embed", "pattern": r"^token_embed\..*"},
        {"type": "module", "name": "pos_embed", "pattern": r"^pos_embed\..*"},
        {"type": "module", "name": "ln", "pattern": r"^ln\..*"},
        {"type": "module", "name": "output_head", "pattern": r"^output_head\..*"},
    ]

    for l in range(n_layers):
        components.append({"type": "module", "name": f"L{l}_attn_qkv", "pattern": f"^transformer\\.layers\\.{l}\\.self_attn\\.in_proj_.*"})
        components.append({"type": "module", "name": f"L{l}_attn_out", "pattern": f"^transformer\\.layers\\.{l}\\.self_attn\\.out_proj\\..*"})
        components.append({"type": "module", "name": f"L{l}_mlp", "pattern": f"^transformer\\.layers\\.{l}\\.linear.*"})
        components.append({"type": "module", "name": f"L{l}_ln1", "pattern": f"^transformer\\.layers\\.{l}\\.norm1\\..*"})
        components.append({"type": "module", "name": f"L{l}_ln2", "pattern": f"^transformer\\.layers\\.{l}\\.norm2\\..*"})
        for h in range(n_heads):
            components.append({
                "type": "head",
                "name": f"L{l}_H{h}",
                "layer_idx": l,
                "head_idx": h
            })

    return components

def freeze_for_rescue(model: nn.Module, comp: Dict[str, Any], d_model: int, n_heads: int):
    """
    Freeze patched parameters so rescue_train only trains the rest of the network.
    """
    if comp["type"] == "module":
        pat = comp["pattern"]
        for name, p in model.named_parameters():
            if re.match(pat, name):
                p.requires_grad_(False)
    elif comp["type"] == "head":
        # For heads, we zero the gradients in the backward pass using a hook on the tensor since it's fused
        # Alternatively, if we just retrain the whole rest of the network, we might freeze the entire self_attn layer.
        # Here we just freeze the entire self_attn as an approximation, or don't freeze the fused weights.
        # Since slicing gradients is complex, we will freeze the whole qkv/out_proj block for simplicity.
        layer_idx = comp["layer_idx"]
        for name, p in model.named_parameters():
            if f"transformer.layers.{layer_idx}.self_attn" in name:
                p.requires_grad_(False)

def build_model(cfg: dict, device: torch.device) -> ModularArithmeticTransformer:
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

def run_transplant(
    donor_dir: Path,
    recipient_dir: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    rescue_steps: int,
    rescue_lr: float,
    direction: str
) -> List[Dict[str, Any]]:

    print(f"[{direction}] donor: {donor_dir} -> recipient: {recipient_dir}")

    donor_model, donor_cfg = load_run(donor_dir, device=device)
    recipient_model, recipient_cfg = load_run(recipient_dir, device=device)

    # Use the recipient's configuration for the dataset to match their environment
    train_loader, test_loader = make_loaders(recipient_cfg, batch_size, device)

    # Baselines
    baseline_donor = evaluate_model(donor_model, train_loader, test_loader, device)
    baseline_recipient = evaluate_model(recipient_model, train_loader, test_loader, device)

    print(f"Donor test acc: {baseline_donor['test_acc']:.4f}")
    print(f"Recipient test acc: {baseline_recipient['test_acc']:.4f}")

    donor_sd = donor_model.state_dict()
    recipient_sd = recipient_model.state_dict()

    d_model = int(recipient_cfg.get("d_model", 128))
    n_heads = int(recipient_cfg.get("n_heads", 4))
    n_layers = int(recipient_cfg.get("n_layers", 1))

    components = get_component_list(d_model, n_heads, n_layers)

    results = []
    results.append({
        "direction": direction,
        "patch": "baseline_donor",
        "test_acc_0_step": baseline_donor["test_acc"],
        "test_acc_k_step": baseline_donor["test_acc"],
    })
    results.append({
        "direction": direction,
        "patch": "baseline_recipient",
        "test_acc_0_step": baseline_recipient["test_acc"],
        "test_acc_k_step": baseline_recipient["test_acc"],
    })

    for comp in tqdm(components, desc=f"Patching {direction}"):
        # Patch
        if comp["type"] == "module":
            patched_sd = patch_module(recipient_sd, donor_sd, comp["pattern"], verbose=False)
            verify_patch(recipient_sd, patched_sd, [comp["pattern"]])
        else:
            patched_sd = patch_attention_head(
                recipient_sd, donor_sd, comp["layer_idx"], comp["head_idx"], d_model, n_heads, verbose=False
            )
            # Cannot easily verify_patch with simple regex for head slices, so we skip it here
            # as our PyTest covers head slices.

        # 0-step
        model_0_step = build_model(recipient_cfg, device)
        model_0_step.load_state_dict(patched_sd)
        res_0 = evaluate_model(model_0_step, train_loader, test_loader, device)

        # K-step
        res_k_acc = res_0["test_acc"]
        if rescue_steps > 0:
            model_k_step = build_model(recipient_cfg, device)
            model_k_step.load_state_dict(patched_sd)
            freeze_for_rescue(model_k_step, comp, d_model, n_heads)

            rescue_wd = recipient_cfg.get("weight_decay", 1.0)
            res_k = rescue_train(
                model_k_step, train_loader, test_loader, device,
                steps=rescue_steps, lr=rescue_lr, weight_decay=rescue_wd
            )
            res_k_acc = res_k["test_acc"]

        # Log
        results.append({
            "direction": direction,
            "patch": comp["name"],
            "test_acc_0_step": res_0["test_acc"],
            "test_acc_k_step": res_k_acc,
            "recovery_0_step": (res_0["test_acc"] - baseline_recipient["test_acc"]) / max(1e-5, baseline_donor["test_acc"] - baseline_recipient["test_acc"]),
            "recovery_k_step": (res_k_acc - baseline_recipient["test_acc"]) / max(1e-5, baseline_donor["test_acc"] - baseline_recipient["test_acc"])
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Component-wise head transplant matrix")
    parser.add_argument("--donor-runs", type=str, nargs="+", required=True, help="List of donor run directories")
    parser.add_argument("--recipient-runs", type=str, nargs="+", required=True, help="List of recipient run directories")
    parser.add_argument("--output-dir", type=str, default="analysis/head_transplant", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rescue-steps", type=int, default=0, help="Steps for fine-tuning after patching")
    parser.add_argument("--rescue-lr", type=float, default=1e-3, help="Learning rate for rescue training")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    all_results = []

    for donor_dir_str, recipient_dir_str in zip(args.donor_runs, args.recipient_runs):
        donor_dir = Path(donor_dir_str)
        recipient_dir = Path(recipient_dir_str)

        # Pure -> Contam
        res_p2c = run_transplant(
            donor_dir=donor_dir,
            recipient_dir=recipient_dir,
            out_dir=out_dir,
            device=device,
            batch_size=args.batch_size,
            rescue_steps=args.rescue_steps,
            rescue_lr=args.rescue_lr,
            direction="pure_to_contam"
        )
        for r in res_p2c:
            r["donor_run"] = donor_dir_str
            r["recipient_run"] = recipient_dir_str
        all_results.extend(res_p2c)

        # Contam -> Pure
        res_c2p = run_transplant(
            donor_dir=recipient_dir, # Swapped!
            recipient_dir=donor_dir, # Swapped!
            out_dir=out_dir,
            device=device,
            batch_size=args.batch_size,
            rescue_steps=args.rescue_steps,
            rescue_lr=args.rescue_lr,
            direction="contam_to_pure"
        )
        for r in res_c2p:
            r["donor_run"] = recipient_dir_str
            r["recipient_run"] = donor_dir_str
        all_results.extend(res_c2p)

    # Save all results
    with open(out_dir / "matrix_results.json", "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)

    print(f"Results saved to {out_dir / 'matrix_results.json'}")

if __name__ == "__main__":
    main()
