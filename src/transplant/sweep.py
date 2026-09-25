import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.transplant.transplant_rescue import (
    load_run, build_model, make_loaders, evaluate_model, rescue_train
)
from src.transplant.patching import patch_head_state_dict, patch_layer_component_state_dict

@dataclass
class ComponentResult:
    component_name: str
    layer_idx: int
    head_idx: Optional[int]
    test_acc: float
    train_acc: float
    fourier_concentration: float
    weight_norm: float
    induction_test_acc: Optional[float] = None
    induction_train_acc: Optional[float] = None


def induction_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    unfrozen_patterns: List[str]
) -> Dict[str, float]:
    """Freeze the entire model except for parameters matching unfrozen_patterns."""
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze matching
    unfrozen_count = 0
    for name, p in model.named_parameters():
        for pat in unfrozen_patterns:
            if re.search(pat, name):
                p.requires_grad_(True)
                unfrozen_count += 1
                break

    if unfrozen_count == 0:
        print("[warn] No parameters were unfrozen for induction training.")
        return evaluate_model(model, train_loader, test_loader, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    it = iter(train_loader)
    model.train()

    # We will use this to zero gradients for the un-transplanted heads if needed.
    # We'll handle this purely by matching patterns to entire tensors here,
    # and if we're doing head-level induction, we assume the user unfreezes the whole QKV matrix
    # for simplicity, as PyTorch doesn't natively support slice-level requires_grad.

    history = []
    for s in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()

        opt.step()

        if s % 10 == 0 or s == steps - 1:
            ev = evaluate_model(model, train_loader, test_loader, device)
            history.append({"step": s, **ev})

    return history[-1], history

def main():
    ap = argparse.ArgumentParser(description="Component-level circuit transplant sweep")
    ap.add_argument("--pure-run", type=Path, required=True)
    ap.add_argument("--contam-run", type=Path, required=True)
    ap.add_argument("--pure-step", type=int, default=None)
    ap.add_argument("--contam-step", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/transplant_sweep"))
    ap.add_argument("--induction-steps", type=int, default=0, help="Steps for targeted induction probing")
    ap.add_argument("--induction-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pure_sd, pure_cfg = load_run(args.pure_run, args.pure_step)
    contam_sd, contam_cfg = load_run(args.contam_run, args.contam_step)

    if int(pure_cfg.get("seed", -1)) != int(contam_cfg.get("seed", -2)):
        print("[warn] seeds differ — train/test split will not match.")

    train_loader, test_loader = make_loaders(contam_cfg, device=device)
    n_layers = int(pure_cfg.get("n_layers", 1))
    n_heads = int(pure_cfg.get("n_heads", 4))
    d_model = int(pure_cfg.get("d_model", 128))
    d_head = d_model // n_heads

    results: List[ComponentResult] = []
    induction_curves = {}

    # 1. Baseline
    print("[run] baseline_contam …")
    base_model = build_model(contam_cfg, device)
    base_model.load_state_dict(contam_sd)
    base_eval = evaluate_model(base_model, train_loader, test_loader, device)

    print("[run] baseline_pure …")
    pure_model = build_model(pure_cfg, device)
    pure_model.load_state_dict(pure_sd)
    pure_eval = evaluate_model(pure_model, train_loader, test_loader, device)

    results.append(ComponentResult("baseline_contam", -1, None, base_eval["test_acc"], base_eval["train_acc"], base_eval["fourier_concentration"], base_eval["weight_norm"]))
    results.append(ComponentResult("baseline_pure", -1, None, pure_eval["test_acc"], pure_eval["train_acc"], pure_eval["fourier_concentration"], pure_eval["weight_norm"]))

    # 2. Iterate over heads
    for l in range(n_layers):
        for h in range(n_heads):
            print(f"[run] transplant Layer {l} Head {h} ...")
            patched_sd = patch_head_state_dict(contam_sd, pure_sd, l, h, d_head, n_heads)
            model = build_model(contam_cfg, device)
            model.load_state_dict(patched_sd)
            ev = evaluate_model(model, train_loader, test_loader, device)

            ind_test, ind_train = None, None
            if args.induction_steps > 0:
                print(f"[run] induction probe Layer {l} Head {h} ...")
                # We unfreeze the whole in_proj / out_proj for this layer as a proxy
                unfrozen_patterns = [
                    f"transformer\\.layers\\.{l}\\.self_attn\\.in_proj_",
                    f"transformer\\.layers\\.{l}\\.self_attn\\.out_proj\\.weight"
                ]
                ind_ev, ind_hist = induction_train(model, train_loader, test_loader, device, args.induction_steps, args.induction_lr, float(contam_cfg.get("weight_decay", 1.0)), unfrozen_patterns)
                ind_test, ind_train = ind_ev["test_acc"], ind_ev["train_acc"]
                induction_curves[f"L{l}_H{h}"] = ind_hist

            results.append(ComponentResult(f"L{l}_H{h}", l, h, ev["test_acc"], ev["train_acc"], ev["fourier_concentration"], ev["weight_norm"], ind_test, ind_train))

    # 3. Iterate over MLPs
    for l in range(n_layers):
        print(f"[run] transplant Layer {l} MLP ...")
        patched_sd = patch_layer_component_state_dict(contam_sd, pure_sd, l, "mlp")
        model = build_model(contam_cfg, device)
        model.load_state_dict(patched_sd)
        ev = evaluate_model(model, train_loader, test_loader, device)

        ind_test, ind_train = None, None
        if args.induction_steps > 0:
            print(f"[run] induction probe Layer {l} MLP ...")
            unfrozen_patterns = [f"transformer\\.layers\\.{l}\\.linear"]
            ind_ev, ind_hist = induction_train(model, train_loader, test_loader, device, args.induction_steps, args.induction_lr, float(contam_cfg.get("weight_decay", 1.0)), unfrozen_patterns)
            ind_test, ind_train = ind_ev["test_acc"], ind_ev["train_acc"]
            induction_curves[f"L{l}_MLP"] = ind_hist

        results.append(ComponentResult(f"L{l}_MLP", l, None, ev["test_acc"], ev["train_acc"], ev["fourier_concentration"], ev["weight_norm"], ind_test, ind_train))

    # Sort results by test accuracy descending
    results_sorted = sorted(results, key=lambda x: x.test_acc, reverse=True)

    # Dump JSON
    json_path = args.output_dir / "sweep_results.json"
    with json_path.open("w") as f:
        json.dump([asdict(r) for r in results_sorted], f, indent=2)

    if args.induction_steps > 0:
        curves_path = args.output_dir / "induction_curves.json"
        with curves_path.open("w") as f:
            json.dump(induction_curves, f, indent=2)

    # Dump Markdown table
    md_path = args.output_dir / "sweep_ranking.md"
    with md_path.open("w") as f:
        f.write("# Component Transplant Sweep Rankings\n\n")
        f.write("| Component | Test Acc | Train Acc | Fourier Conc | Norm | Induction Test Acc |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results_sorted:
            ind_test = f"{r.induction_test_acc:.3f}" if r.induction_test_acc is not None else "N/A"
            f.write(f"| {r.component_name} | {r.test_acc:.3f} | {r.train_acc:.3f} | {r.fourier_concentration:.3f} | {r.weight_norm:.2f} | {ind_test} |\n")

    print(f"\n[done] Ranked component importance saved to {md_path}")

if __name__ == "__main__":
    main()
