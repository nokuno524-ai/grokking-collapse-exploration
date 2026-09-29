import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

from ..checkpoint import load_checkpoint
from ..model import ModularArithmeticTransformer
from .patching import patch_state_dict, get_keys_for_component
from ..train import evaluate
from ..data import DatasetConfig, generate_modular_arithmetic
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def build_model(cfg: dict, device: torch.device) -> ModularArithmeticTransformer:
    """Build a new ModularArithmeticTransformer based on a config dictionary."""
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)

def make_loaders(cfg: dict, batch_size: int = 512, device: torch.device = torch.device("cpu")):
    """Create train and test DataLoaders for modular arithmetic based on config."""
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

def evaluate_model_full(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate a model on train and test loaders, return metrics including loss gap."""
    train_loss, train_acc = evaluate(model, train_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "loss_gap": test_loss - train_loss
    }

def induction_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    component_keys: List[str],
    steps: int = 1000,
    lr: float = 1e-3,
    wd: float = 1.0
) -> float:
    """Fine-tune only the specified component keys and return final test accuracy."""

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze transplanted
    for name, p in model.named_parameters():
        if name in component_keys:
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return evaluate_model_full(model, train_loader, test_loader, device)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    model.train()

    it = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    return evaluate_model_full(model, train_loader, test_loader, device)

import argparse
from pathlib import Path
import json
import logging
from typing import List, Tuple



def find_seed_pairs(pure_dir: Path, contam_dir: Path) -> List[Tuple[Path, Path]]:
    pure_runs = list(pure_dir.glob("seed_*"))
    contam_runs = list(contam_dir.glob("seed_*"))

    pairs = []
    for pr in pure_runs:
        seed_name = pr.name
        cr = contam_dir / seed_name
        if cr.exists():
            pairs.append((pr, cr))
            if len(pairs) >= 3:
                break
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-dir", type=Path, required=True, help="Directory containing pure seeds")
    parser.add_argument("--contam-dir", type=Path, required=True, help="Directory containing contam seeds")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/ablation_grid"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = find_seed_pairs(args.pure_dir, args.contam_dir)
    if not pairs:
        print("No seed pairs found.")
        return

    from src.transplant.patching import COMPONENT_PATTERNS, patch_head_state_dict
    from src.transplant.atlas import zero_ablate_component, mean_ablate_component, resample_ablate_component

    results = []

    for pure_run, contam_run in pairs:
        print(f"Processing pair: {pure_run.name}")
        pure_sd, pure_cfg = load_checkpoint(pure_run)
        contam_sd, contam_cfg = load_checkpoint(contam_run)

        train_loader, test_loader = make_loaders(contam_cfg, device=device)

        def eval_sd(sd, name, component_name, type_str):
            model = build_model(contam_cfg, device)
            model.load_state_dict(sd, strict=False)
            metrics = evaluate_model_full(model, train_loader, test_loader, device)
            wn = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5

            results.append({
                "seed": pure_run.name,
                "variant": name,
                "component": component_name,
                "type": type_str,
                "metrics": metrics,
                "weight_norm": float(wn)
            })

        # Baselines
        eval_sd(pure_sd, "baseline_pure", "all", "baseline")
        eval_sd(contam_sd, "baseline_contam", "none", "baseline")

        # Individual layer weights
        components = list(pure_sd.keys())
        for comp in components:
            if comp not in contam_sd:
                continue

            # pure -> contam
            patched_p2c = {k: v.clone() for k, v in contam_sd.items()}
            patched_p2c[comp] = pure_sd[comp].clone()
            eval_sd(patched_p2c, f"p2c_{comp}", comp, "transplant_p2c")

            # contam -> pure
            patched_c2p = {k: v.clone() for k, v in pure_sd.items()}
            patched_c2p[comp] = contam_sd[comp].clone()
            eval_sd(patched_c2p, f"c2p_{comp}", comp, "transplant_c2p")

            # Ablations on pure
            zero_p = zero_ablate_component(pure_sd, [comp])
            eval_sd(zero_p, f"zero_pure_{comp}", comp, "zero")

            mean_p = mean_ablate_component(pure_sd, [comp])
            eval_sd(mean_p, f"mean_pure_{comp}", comp, "mean")

            res_p = resample_ablate_component(pure_sd, [comp])
            eval_sd(res_p, f"resample_pure_{comp}", comp, "resample")

        # Attention heads
        n_layers = contam_cfg.get("n_layers", 1)
        n_heads = contam_cfg.get("n_heads", 4)
        d_model = contam_cfg.get("d_model", 128)

        for l in range(n_layers):
            for h in range(n_heads):
                head_name = f"layer_{l}_head_{h}"

                # pure -> contam head
                patched_head_p2c = patch_head_state_dict(contam_sd, pure_sd, l, h, d_model, n_heads)
                eval_sd(patched_head_p2c, f"p2c_{head_name}", head_name, "transplant_head_p2c")

                # contam -> pure head
                patched_head_c2p = patch_head_state_dict(pure_sd, contam_sd, l, h, d_model, n_heads)
                eval_sd(patched_head_c2p, f"c2p_{head_name}", head_name, "transplant_head_c2p")

        # Combination: all linear1 and linear2 (MLP)
        mlp_keys = [k for k in pure_sd.keys() if "linear1" in k or "linear2" in k]

        patched_mlp_p2c = {k: v.clone() for k, v in contam_sd.items()}
        for k in mlp_keys: patched_mlp_p2c[k] = pure_sd[k].clone()
        eval_sd(patched_mlp_p2c, "p2c_mlp_all", "mlp_all", "transplant_combo_p2c")

        patched_mlp_c2p = {k: v.clone() for k, v in pure_sd.items()}
        for k in mlp_keys: patched_mlp_c2p[k] = contam_sd[k].clone()
        eval_sd(patched_mlp_c2p, "c2p_mlp_all", "mlp_all", "transplant_combo_c2p")

    out_path = args.output_dir / "ablation_grid_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
