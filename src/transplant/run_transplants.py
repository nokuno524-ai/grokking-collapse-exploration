"""
Experiment harness to run a grid of circuit transplant experiments.
Evaluates swapping different components (heads, MLP, LN) from a donor
checkpoint into a recipient checkpoint.
"""

import argparse
import csv
import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from torch.utils.data import TensorDataset, DataLoader
from src.transplant.circuits import swap_attention_head, swap_mlp, swap_layer_norm
from src.transplant_rescue import load_run

@dataclass
class TransplantResult:
    donor: str
    recipient: str
    component: str
    granularity: str
    zero_shot_acc: float
    finetune_acc: Optional[float]
    test_loss: float

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction='sum')
            preds = logits.argmax(dim=-1)
            total_loss += loss.item()
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples

def make_loader(cfg, device):
    dc = DatasetConfig(
        prime=int(cfg.get("prime", 59)),
        train_fraction=float(cfg.get("train_fraction", 0.3)),
        collapse_level=float(cfg.get("collapse_level", 0.0)),
        collapse_severity=float(cfg.get("collapse_severity", 0.0)),
        noise_fraction=float(cfg.get("noise_fraction", 0.0)),
        seed=int(cfg.get("seed", 42)),
    )
    _, _, test_in, test_tgt = generate_modular_arithmetic(dc)
    ds = TensorDataset(test_in, test_tgt)
    return DataLoader(ds, batch_size=512, shuffle=False)

def build_model(cfg, device):
    m = ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    )
    m.to(device)
    return m

def finetune(model, cfg, loader, device, frozen_keys, steps=100, lr=1e-3):
    model.train()

    for name, p in model.named_parameters():
        if name in frozen_keys:
            p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Simple unrolled training loop for a few steps
    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-runs", nargs="+", required=True)
    parser.add_argument("--recipient-runs", nargs="+", required=True)
    parser.add_argument("--donor-step", type=int, default=None)
    parser.add_argument("--recipient-step", type=int, default=None)
    parser.add_argument("--components", nargs="+", default=["heads", "mlp", "ln"])
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/transplants.csv"))
    parser.add_argument("--finetune-steps", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for d_path in args.donor_runs:
        for r_path in args.recipient_runs:
            d_path = Path(d_path)
            r_path = Path(r_path)
            try:
                donor_sd, donor_cfg = load_run(d_path, args.donor_step)
                recip_sd, recip_cfg = load_run(r_path, args.recipient_step)
            except FileNotFoundError as e:
                print(f"Skipping pair due to missing files: {e}")
                continue

            test_loader = make_loader(recip_cfg, device)
            n_layers = int(recip_cfg.get("n_layers", 1))
            n_heads = int(recip_cfg.get("n_heads", 4))
            d_model = int(recip_cfg.get("d_model", 128))

            for comp in args.components:
                variants = []
                if comp == "heads":
                    # Single heads
                    for l in range(n_layers):
                        for h in range(n_heads):
                            sd = swap_attention_head(recip_sd, donor_sd, l, h, d_model, n_heads)
                            keys = {
                                f"transformer.layers.{l}.self_attn.in_proj_weight",
                                f"transformer.layers.{l}.self_attn.in_proj_bias",
                                f"transformer.layers.{l}.self_attn.out_proj.weight"
                            }
                            variants.append((f"L{l}_H{h}", sd, keys))
                    # Full heads (all)
                    full_sd = recip_sd.copy()
                    keys = set()
                    for l in range(n_layers):
                        for h in range(n_heads):
                            full_sd = swap_attention_head(full_sd, donor_sd, l, h, d_model, n_heads)
                            keys.update({
                                f"transformer.layers.{l}.self_attn.in_proj_weight",
                                f"transformer.layers.{l}.self_attn.in_proj_bias",
                                f"transformer.layers.{l}.self_attn.out_proj.weight"
                            })
                    variants.append(("full", full_sd, keys))
                elif comp == "mlp":
                    for l in range(n_layers):
                        sd = swap_mlp(recip_sd, donor_sd, l)
                        keys = {
                            f"transformer.layers.{l}.linear1.weight",
                            f"transformer.layers.{l}.linear1.bias",
                            f"transformer.layers.{l}.linear2.weight",
                            f"transformer.layers.{l}.linear2.bias"
                        }
                        variants.append((f"L{l}", sd, keys))
                elif comp == "ln":
                    for l in range(n_layers):
                        sd = swap_layer_norm(recip_sd, donor_sd, l)
                        keys = {
                            f"transformer.layers.{l}.norm1.weight",
                            f"transformer.layers.{l}.norm1.bias",
                            f"transformer.layers.{l}.norm2.weight",
                            f"transformer.layers.{l}.norm2.bias"
                        }
                        variants.append((f"L{l}", sd, keys))

                for granularity, patched_sd, frozen_keys in variants:
                    model = build_model(recip_cfg, device)
                    # Use strict=True as per task requirement (we fix the failure mode)
                    model.load_state_dict(patched_sd, strict=True)

                    z_loss, z_acc = evaluate(model, test_loader, device)

                    ft_acc = None
                    if args.finetune_steps > 0:
                        finetune(model, recip_cfg, test_loader, device, frozen_keys, steps=args.finetune_steps)
                        _, ft_acc = evaluate(model, test_loader, device)

                    results.append(TransplantResult(
                        donor=d_path.name,
                        recipient=r_path.name,
                        component=comp,
                        granularity=granularity,
                        zero_shot_acc=z_acc,
                        finetune_acc=ft_acc,
                        test_loss=z_loss
                    ))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["donor", "recipient", "component", "granularity", "zero_shot_acc", "finetune_acc", "test_loss"])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"Saved {len(results)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
