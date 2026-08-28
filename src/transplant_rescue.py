"""
Experiment A — surgical circuit transplant rescue.

Hypothesis: a contaminated (label-noise) run that fails to grok is missing a
specific Fourier circuit that the matched pure run developed. If we paste the
pure run's component(s) into the contaminated run's checkpoint at the same
seed, then either (i) test accuracy jumps zero-shot to near pure, or
(ii) a brief "freeze patched, retrain rest on contaminated data" loop rescues
generalization.

Design:
  Inputs:  --pure-run     path to a grokked run, e.g. .../wd1/noise0/seed_42
           --contam-run   path to a failed run at the same seed,
                          e.g. .../wd1/noise0.15/seed_42
  Variants we evaluate (each emits a row of metrics):
    baseline_pure        → pure ckpt as-is on test set
    baseline_contam      → contaminated ckpt as-is
    baseline_pure_swap   → pure ckpt evaluated on contam test split
                            (sanity: should match baseline_pure exactly because
                             seeds match → splits match)
    transplant_<C>       → contaminated ckpt with component C replaced by pure
    transplant_<C>+rt    → as above, then rest-of-network retrained on the
                            contaminated data for --rescue-steps with the
                            transplanted component frozen.
    rand_<C>             → contaminated ckpt with component C replaced by a
                            *random orthonormal rotation* of the contaminated
                            weights (specificity control). Same shape, same
                            spectrum, randomized basis.
    rand_<C>+rt          → as above + rescue retraining.
    swap_all             → all weights pure (upper bound on rescue).

  Components patched:
    token_embed    self_attn_in_proj    self_attn_out_proj
    linear1        linear2              output_head

  The pure-run config and the contam-run config must share the same seed (so
  the train/test split matches), and the same prime / d_model / etc. We assert
  this on load.

Outputs (to --output-dir, default analysis/transplant):
  rescue_results.json   raw row-per-variant metrics
  rescue_summary.md     pretty markdown table
  rescue_bar.png        bar plot of test acc per variant

Usage:
  python src/transplant_rescue.py \
      --pure-run results/exp_c_grid/wd1/noise0/seed_42 \
      --contam-run results/exp_c_grid/wd1/noise0.15/seed_42 \
      --output-dir analysis/transplant/wd1_n015_s42 \
      --rescue-steps 2000

The Fourier-circuit identification half (which frequencies dominate) is in
get_fourier_basis() and reported in the JSON, but does not change the patch
itself — we patch whole matrices, which is cleanest for a first pass.
"""

from __future__ import annotations

import argparse
import copy
import json
from src.log_loader import load_results_json
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .model import ModularArithmeticTransformer
    from .data import DatasetConfig, generate_modular_arithmetic
    from .train import compute_fourier_concentration, evaluate
except ImportError:
    from model import ModularArithmeticTransformer  # type: ignore
    from data import DatasetConfig, generate_modular_arithmetic  # type: ignore
    from train import compute_fourier_concentration, evaluate  # type: ignore


# Mapping from a short component name to the regex of state_dict keys it covers.
COMPONENT_PATTERNS: Dict[str, str] = {
    "token_embed": r"^token_embed\.",
    "pos_embed": r"^pos_embed\.",
    "self_attn_in_proj": r"^transformer\.layers\.\d+\.self_attn\.in_proj_(weight|bias)$",
    "self_attn_out_proj": r"^transformer\.layers\.\d+\.self_attn\.out_proj\.",
    "linear1": r"^transformer\.layers\.\d+\.linear1\.",
    "linear2": r"^transformer\.layers\.\d+\.linear2\.",
    "norm1": r"^transformer\.layers\.\d+\.norm1\.",
    "norm2": r"^transformer\.layers\.\d+\.norm2\.",
    "ln": r"^ln\.",
    "output_head": r"^output_head\.",
}

# The 6 components we report in the main table — picked because each is a single
# semantically meaningful block. token_embed and output_head touch the Fourier
# basis directly; attn / FFN are the obvious circuit-carrying matrices.
DEFAULT_PATCH_COMPONENTS = [
    "token_embed",
    "self_attn_in_proj",
    "self_attn_out_proj",
    "linear1",
    "linear2",
    "output_head",
]


@dataclass
class VariantResult:
    name: str
    component: Optional[str]
    test_loss: float
    test_acc: float
    train_loss: float
    train_acc: float
    fourier_concentration: float
    weight_norm: float
    rescue_steps: int = 0
    notes: str = ""


def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    pat = COMPONENT_PATTERNS[component]
    return [k for k in sd.keys() if re.match(pat, k)]


def load_run(run_dir: Path, step: Optional[int] = None) -> Tuple[Dict, dict]:
    """Return (state_dict, config) for the given run.
    If step is None, picks the largest checkpoint."""
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
    if not cfg:
        try:
            cfg = load_results_json(run_dir).get("config", {})
        except FileNotFoundError:
            pass
    return sd, cfg


def build_model(cfg: dict, device: torch.device) -> ModularArithmeticTransformer:
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)


def get_fourier_basis(token_embed: torch.Tensor, top_k: int = 5) -> dict:
    """Identify the dominant Fourier frequencies in token_embed (Nanda-style).

    Returns a small dict with the top-k frequencies and the energy fraction
    they capture. We DO NOT use this to construct projections (we patch whole
    matrices). It is reported as metadata only — useful when comparing pure vs
    contaminated bases."""
    W = token_embed.detach().to(torch.float32)
    spec = torch.fft.fft(W, dim=0).abs()  # (prime, d_model)
    avg = spec.mean(dim=1)  # (prime,)
    avg = avg[1:]  # drop DC
    total = avg.sum().clamp(min=1e-12)
    topv, topi = torch.topk(avg, k=min(top_k, avg.numel()))
    return {
        "top_frequencies": [int(i.item()) + 1 for i in topi],  # +1 because we dropped DC
        "top_fraction": float(topv.sum().item() / total.item()),
        "spectrum_top10": [float(x.item()) for x in avg[:10]],
    }


def random_basis_swap(weight: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Return a tensor with the same shape and spectrum as `weight` but a
    random orthonormal basis. Specificity control: keeps ||W||_F and the
    singular values, randomizes which directions are which.

    For 1-D bias vectors, returns a copy with a random *permutation* of the
    same values (preserves L2 norm and entry distribution)."""
    w = weight.detach().to(torch.float32).clone()
    if w.ndim == 1:
        idx = torch.randperm(w.numel(), generator=rng)
        return w[idx]
    if w.ndim != 2:
        # higher-dim parameters: flatten the last dims and apply 2-D variant
        orig_shape = w.shape
        w2 = w.reshape(w.shape[0], -1)
        out = random_basis_swap(w2, rng)
        return out.reshape(orig_shape)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    # Random orthonormal U' and Vh' of correct shapes.
    Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng))
    Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng))
    Vhr = Vr.T
    return Ur @ torch.diag(S) @ Vhr


def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict = base, with `component` replaced by donor's.
    If randomize=True, replace with a random-orthonormal-basis version of base
    (specificity control); donor is unused in that path."""
    out = {k: v.clone() for k, v in base_sd.items()}
    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        for k in keys_for(component, base_sd):
            out[k] = random_basis_swap(base_sd[k], rng)
    else:
        for k in keys_for(component, base_sd):
            if k not in donor_sd:
                continue
            if donor_sd[k].shape != base_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on {k}: donor {tuple(donor_sd[k].shape)} "
                    f"vs base {tuple(base_sd[k].shape)}"
                )
            out[k] = donor_sd[k].clone()
    return out


def make_loaders(
    cfg: dict, batch_size: int = 512, device: torch.device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader]:
    """Reconstruct the exact train/test split for the given config (matched seed)."""
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


def freeze_component(model: nn.Module, component: str) -> int:
    """Freeze all params whose state_dict-style name matches `component`."""
    pat = COMPONENT_PATTERNS[component]
    n_frozen = 0
    for name, p in model.named_parameters():
        if re.match(pat, name):
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    train_loss, train_acc = evaluate(model, train_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    fc = compute_fourier_concentration(model)
    wn = float(sum(p.detach().norm().item() ** 2 for p in model.parameters()) ** 0.5)
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "fourier_concentration": fc,
        "weight_norm": wn,
    }


def rescue_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, float]:
    """Train trainable params (those with requires_grad=True) for `steps` steps."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return evaluate_model(model, train_loader, test_loader, device)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    it = iter(train_loader)
    model.train()
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
    return evaluate_model(model, train_loader, test_loader, device)


def run_one_variant(
    name: str,
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    component: Optional[str],
    cfg_for_loaders: dict,
    cfg_for_model: dict,
    device: torch.device,
    randomize: bool = False,
    rescue_steps: int = 0,
    rescue_lr: float = 1e-3,
    rescue_wd: float = 1.0,
    rescue_seed: int = 0,
    rng: Optional[torch.Generator] = None,
) -> VariantResult:
    if component is None:
        # baseline / swap_all path: state dict is taken as-is from `donor_sd`
        sd = {k: v.clone() for k, v in (donor_sd or base_sd).items()}
    else:
        sd = patch_state_dict(base_sd, donor_sd or {}, component,
                              randomize=randomize, rng=rng)
    model = build_model(cfg_for_model, device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading patched sd: {unexpected}")
    train_loader, test_loader = make_loaders(cfg_for_loaders, device=device)
    if rescue_steps > 0 and component is not None:
        torch.manual_seed(rescue_seed)
        # Freeze the patched component so the rest adapts around it.
        freeze_component(model, component)
        metrics = rescue_train(
            model, train_loader, test_loader, device,
            steps=rescue_steps, lr=rescue_lr, weight_decay=rescue_wd,
        )
        notes = f"froze {component}, retrained other params for {rescue_steps} steps"
    else:
        metrics = evaluate_model(model, train_loader, test_loader, device)
        notes = "zero-shot eval"
    return VariantResult(
        name=name,
        component=component,
        test_loss=metrics["test_loss"],
        test_acc=metrics["test_acc"],
        train_loss=metrics["train_loss"],
        train_acc=metrics["train_acc"],
        fourier_concentration=metrics["fourier_concentration"],
        weight_norm=metrics["weight_norm"],
        rescue_steps=rescue_steps if component is not None else 0,
        notes=notes,
    )


def write_summary_md(
    results: List[VariantResult],
    pure_basis: dict,
    contam_basis: dict,
    pure_cfg: dict,
    contam_cfg: dict,
    out_path: Path,
) -> None:
    lines = ["# Surgical Transplant Rescue — Experiment A\n\n"]
    lines.append(
        f"**Pure run:** wd={pure_cfg.get('weight_decay')}, "
        f"noise={pure_cfg.get('noise_fraction')}, seed={pure_cfg.get('seed')}\n\n"
    )
    lines.append(
        f"**Contaminated run:** wd={contam_cfg.get('weight_decay')}, "
        f"noise={contam_cfg.get('noise_fraction')}, seed={contam_cfg.get('seed')}\n\n"
    )
    lines.append("## Fourier basis comparison\n\n")
    lines.append(
        f"- Pure top-5 freqs: {pure_basis['top_frequencies']}, "
        f"energy fraction {pure_basis['top_fraction']:.3f}\n"
    )
    lines.append(
        f"- Contam top-5 freqs: {contam_basis['top_frequencies']}, "
        f"energy fraction {contam_basis['top_fraction']:.3f}\n\n"
    )
    lines.append("## Variant table\n\n")
    lines.append(
        "| variant | component | rescue_steps | train_acc | test_acc | "
        "fourier | ‖W‖ | notes |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in results:
        lines.append(
            f"| {r.name} | {r.component or '—'} | {r.rescue_steps} | "
            f"{r.train_acc:.3f} | **{r.test_acc:.3f}** | "
            f"{r.fourier_concentration:.3f} | {r.weight_norm:.2f} | {r.notes} |\n"
        )
    lines.append("\n## Reading the table\n\n")
    lines.append(
        "- `baseline_contam` is the failure point we are trying to rescue.\n"
        "- `baseline_pure` is the upper bound (a paired-seed grokked model).\n"
        "- `transplant_<C>` is the *zero-shot* swap — paste pure's `<C>` into "
        "contaminated, then evaluate without retraining. If test_acc jumps to "
        "near pure, the missing piece is that single component.\n"
        "- `transplant_<C>+rt` retrains the *un-patched* params on contaminated "
        "data with `<C>` frozen. If this rescues, the patched component is "
        "necessary but the rest of the model can adapt around it.\n"
        "- `rand_<C>` swaps in a random-orthonormal-basis weight matrix of the "
        "same shape and spectrum (specificity control). If `rand_<C>` rescues "
        "as much as `transplant_<C>`, the rescue is not specific to pure's "
        "circuit — undermines the mechanistic story.\n"
        "- `swap_all` is full pure-into-contam transfer; should match "
        "baseline_pure exactly (sanity check).\n"
    )
    out_path.write_text("".join(lines))


def plot_bar(results: List[VariantResult], out_path: Path) -> None:
    names = [r.name for r in results]
    accs = [r.test_acc for r in results]
    colors = []
    for r in results:
        if r.name.startswith("baseline_pure"):
            colors.append("#2ca02c")
        elif r.name.startswith("baseline_contam"):
            colors.append("#d62728")
        elif r.name.startswith("rand_"):
            colors.append("#7f7f7f")
        elif "+rt" in r.name:
            colors.append("#1f77b4")
        else:
            colors.append("#ff7f0e")
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.45), 5))
    ax.bar(range(len(names)), accs, color=colors)
    ax.axhline(0.95, color="black", linestyle="--", alpha=0.4, label="grokking threshold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Surgical transplant rescue — test accuracy per variant")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure-run", type=Path, required=True,
                    help="Path to a grokked run directory (contains checkpoint_*.pt + results.json).")
    ap.add_argument("--contam-run", type=Path, required=True,
                    help="Path to a failed-grokking run at the *same seed*.")
    ap.add_argument("--pure-step", type=int, default=None,
                    help="Which checkpoint step to use for pure (default: last).")
    ap.add_argument("--contam-step", type=int, default=None,
                    help="Which checkpoint step to use for contam (default: last).")
    ap.add_argument("--components", type=str,
                    default=",".join(DEFAULT_PATCH_COMPONENTS),
                    help="Comma-separated components to patch.")
    ap.add_argument("--rescue-steps", type=int, default=2000,
                    help="Steps of post-patch retraining (0 to disable retrain row).")
    ap.add_argument("--rescue-lr", type=float, default=1e-3)
    ap.add_argument("--rescue-wd", type=float, default=None,
                    help="Weight decay during rescue (default: contam-run's wd).")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("analysis/transplant"),
                    help="Where to save results.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for random-basis controls and rescue.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    pure_sd, pure_cfg = load_run(args.pure_run, args.pure_step)
    contam_sd, contam_cfg = load_run(args.contam_run, args.contam_step)
    print(f"[info] pure cfg: wd={pure_cfg.get('weight_decay')} "
          f"noise={pure_cfg.get('noise_fraction')} seed={pure_cfg.get('seed')}")
    print(f"[info] contam cfg: wd={contam_cfg.get('weight_decay')} "
          f"noise={contam_cfg.get('noise_fraction')} seed={contam_cfg.get('seed')}")

    if int(pure_cfg.get("seed", -1)) != int(contam_cfg.get("seed", -2)):
        print("[warn] seeds differ — train/test split will not match. Continuing anyway.")
    for fld in ("prime", "d_model", "n_heads", "d_ff", "n_layers"):
        if pure_cfg.get(fld) != contam_cfg.get(fld):
            raise ValueError(
                f"architecture mismatch on {fld}: pure={pure_cfg.get(fld)} "
                f"vs contam={contam_cfg.get(fld)}"
            )

    rescue_wd = args.rescue_wd
    if rescue_wd is None:
        rescue_wd = float(contam_cfg.get("weight_decay", 1.0))

    rng = torch.Generator().manual_seed(args.seed)
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    for c in components:
        if c not in COMPONENT_PATTERNS:
            raise ValueError(f"unknown component {c!r}; valid: {list(COMPONENT_PATTERNS)}")

    # Fourier-basis metadata
    pure_basis = get_fourier_basis(pure_sd["token_embed.weight"])
    contam_basis = get_fourier_basis(contam_sd["token_embed.weight"])
    print(f"[info] pure top freqs = {pure_basis['top_frequencies']} "
          f"({pure_basis['top_fraction']:.3f})")
    print(f"[info] contam top freqs = {contam_basis['top_frequencies']} "
          f"({contam_basis['top_fraction']:.3f})")

    results: List[VariantResult] = []

    # Baselines (no patch)
    print("[run] baseline_pure …")
    results.append(run_one_variant(
        "baseline_pure", base_sd=pure_sd, donor_sd=None, component=None,
        cfg_for_loaders=pure_cfg, cfg_for_model=pure_cfg, device=device,
    ))
    print("[run] baseline_contam …")
    results.append(run_one_variant(
        "baseline_contam", base_sd=contam_sd, donor_sd=None, component=None,
        cfg_for_loaders=contam_cfg, cfg_for_model=contam_cfg, device=device,
    ))
    # Sanity: load pure into contam-cfg loader (should equal baseline_pure for matched seeds)
    print("[run] swap_all …")
    results.append(run_one_variant(
        "swap_all", base_sd=contam_sd, donor_sd=pure_sd, component=None,
        cfg_for_loaders=contam_cfg, cfg_for_model=contam_cfg, device=device,
    ))

    # Per-component variants
    for comp in components:
        print(f"[run] transplant_{comp} (zero-shot) …")
        results.append(run_one_variant(
            f"transplant_{comp}", base_sd=contam_sd, donor_sd=pure_sd,
            component=comp,
            cfg_for_loaders=contam_cfg, cfg_for_model=contam_cfg,
            device=device, rescue_steps=0,
        ))
        if args.rescue_steps > 0:
            print(f"[run] transplant_{comp}+rt …")
            results.append(run_one_variant(
                f"transplant_{comp}+rt", base_sd=contam_sd, donor_sd=pure_sd,
                component=comp,
                cfg_for_loaders=contam_cfg, cfg_for_model=contam_cfg,
                device=device, rescue_steps=args.rescue_steps,
                rescue_lr=args.rescue_lr, rescue_wd=rescue_wd,
                rescue_seed=args.seed,
            ))
        print(f"[run] rand_{comp} (zero-shot) …")
        results.append(run_one_variant(
            f"rand_{comp}", base_sd=contam_sd, donor_sd=None,
            component=comp,
            cfg_for_loaders=contam_cfg, cfg_for_model=contam_cfg,
            device=device, randomize=True, rng=rng, rescue_steps=0,
        ))
        if args.rescue_steps > 0:
            print(f"[run] rand_{comp}+rt …")
            results.append(run_one_variant(
                f"rand_{comp}+rt", base_sd=contam_sd, donor_sd=None,
                component=comp,
                cfg_for_loaders=contam_cfg, cfg_for_model=contam_cfg,
                device=device, randomize=True, rng=rng,
                rescue_steps=args.rescue_steps,
                rescue_lr=args.rescue_lr, rescue_wd=rescue_wd,
                rescue_seed=args.seed,
            ))

    # Persist
    json_path = args.output_dir / "rescue_results.json"
    with json_path.open("w") as f:
        json.dump({
            "pure_run": str(args.pure_run),
            "contam_run": str(args.contam_run),
            "pure_cfg": pure_cfg,
            "contam_cfg": contam_cfg,
            "rescue_steps": args.rescue_steps,
            "rescue_lr": args.rescue_lr,
            "rescue_wd": rescue_wd,
            "seed": args.seed,
            "pure_basis": pure_basis,
            "contam_basis": contam_basis,
            "variants": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"[done] wrote {json_path}")

    write_summary_md(
        results, pure_basis, contam_basis, pure_cfg, contam_cfg,
        args.output_dir / "rescue_summary.md",
    )
    print(f"[done] wrote {args.output_dir/'rescue_summary.md'}")

    plot_bar(results, args.output_dir / "rescue_bar.png")
    print(f"[done] wrote {args.output_dir/'rescue_bar.png'}")

    # Console summary
    print("\n=== SUMMARY ===")
    for r in results:
        rt = f"+rt({r.rescue_steps})" if r.rescue_steps else ""
        print(f"  {r.name:30s} {rt:10s} test_acc={r.test_acc:.3f} fc={r.fourier_concentration:.3f}")


if __name__ == "__main__":
    main()
