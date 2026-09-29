"""
Systematic circuit transplant sweep.
Extracts attention-head/MLP circuits from checkpoints trained under each condition
and transplants them into models from every other condition.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import product
from contextlib import contextmanager

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.train import evaluate
    from src.transplant.patching import patch_state_dict, patch_head_state_dict
except ImportError:
    from model import ModularArithmeticTransformer  # type: ignore
    from data import DatasetConfig, generate_modular_arithmetic  # type: ignore
    from train import evaluate  # type: ignore
    from transplant.patching import patch_state_dict, patch_head_state_dict  # type: ignore


@contextmanager
def graft_context(model: torch.nn.Module, patched_sd: Dict[str, torch.Tensor]):
    """
    Context manager to safely load a patched state dict, yield, and restore original state.
    """
    orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict(patched_sd, strict=True)
        yield model
    finally:
        model.load_state_dict(orig_sd, strict=True)


def load_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """Load a checkpoint and its config."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}")
    return ckpt


def run_transplant_matrix(
    base_dir: Path,
    components_to_test: List[str],
    conditions: List[str],
    device: torch.device,
    seed: int = 42,
    batch_size: int = 512,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a full condition x condition matrix sweep of circuit transplants.
    """
    results_matrix = []

    # First, load all final checkpoints for the requested conditions.
    # We assume one seed for the pure matrix sweep, or we use the first available.
    models_data = {}
    loaders = {}

    for condition in conditions:
        cond_dir = base_dir / condition
        if not cond_dir.exists():
            print(f"Warning: directory {cond_dir} does not exist, skipping.")
            continue

        # Find the latest checkpoint
        ckpts = sorted(cond_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if not ckpts:
            print(f"Warning: no checkpoints found in {cond_dir}, skipping.")
            continue

        ckpt = load_checkpoint(ckpts[-1])
        cfg = ckpt.get("config", {})
        sd = ckpt["model_state"]
        models_data[condition] = {"sd": sd, "cfg": cfg}

        # Make a dataloader based on this condition's config to test on its test set
        dc = DatasetConfig(
            prime=int(cfg.get("prime", 59)),
            train_fraction=float(cfg.get("train_fraction", 0.3)),
            collapse_level=float(cfg.get("collapse_level", 0.0)),
            collapse_severity=float(cfg.get("collapse_severity", 0.5)),
            noise_fraction=float(cfg.get("noise_fraction", 0.0)),
            seed=int(cfg.get("seed", seed)),
        )
        _, _, test_in, test_tgt = generate_modular_arithmetic(dc)
        test_ds = torch.utils.data.TensorDataset(test_in, test_tgt)
        loaders[condition] = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    valid_conditions = list(models_data.keys())

    # Initialize recipient models once
    recipient_models = {}
    for target_cond in valid_conditions:
        cfg = models_data[target_cond]["cfg"]
        model = ModularArithmeticTransformer(
            prime=int(cfg.get("prime", 59)),
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            d_ff=int(cfg.get("d_ff", 512)),
            n_layers=int(cfg.get("n_layers", 1)),
        ).to(device)
        model.load_state_dict(models_data[target_cond]["sd"])
        # Important: set eval mode to ensure deterministic evaluations
        model.eval()
        recipient_models[target_cond] = model

    # Evaluate baselines (untouched models)
    baselines = {}
    for cond in valid_conditions:
        model = recipient_models[cond]
        loss, acc = evaluate(model, loaders[cond], device)
        baselines[cond] = {"test_loss": loss, "test_acc": acc}

    # Run the matrix: Source (Donor) x Target (Recipient)
    for source_cond in valid_conditions:
        for target_cond in valid_conditions:
            for component in components_to_test:
                source_sd = models_data[source_cond]["sd"]
                target_sd = models_data[target_cond]["sd"]

                # Apply the patch
                # Support head-level patches like "head_0_0" -> layer 0, head 0
                if component.startswith("head_"):
                    parts = component.split("_")
                    layer_idx = int(parts[1])
                    head_idx = int(parts[2])
                    cfg = models_data[target_cond]["cfg"]
                    d_model = int(cfg.get("d_model", 128))
                    n_heads = int(cfg.get("n_heads", 4))
                    patched_sd = patch_head_state_dict(
                        target_sd, source_sd, layer_idx, head_idx, d_model, n_heads
                    )
                else:
                    patched_sd = patch_state_dict(target_sd, source_sd, component)

                target_model = recipient_models[target_cond]

                # Use context manager to safely inject and test
                with graft_context(target_model, patched_sd) as patched_model:
                    loss, acc = evaluate(patched_model, loaders[target_cond], device)

                results_matrix.append({
                    "source": source_cond,
                    "target": target_cond,
                    "component": component,
                    "test_loss": loss,
                    "test_acc": acc,
                    "acc_diff": acc - baselines[target_cond]["test_acc"]
                })
                print(f"Donor={source_cond:15} | Recipient={target_cond:15} | Comp={component:12} | Acc={acc:.4f} (Diff={acc - baselines[target_cond]['test_acc']:.4f})")

    output = {
        "baselines": baselines,
        "results": results_matrix
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("analysis/transplant_matrix.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default sweep over main conditions
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    components = [
        "self_attn_in_proj", "self_attn_out_proj",
        "linear1", "linear2",
        "head_0_0", "head_0_1", "head_0_2", "head_0_3"
    ]

    run_transplant_matrix(args.base_dir, components, conditions, device, output_file=args.output)
