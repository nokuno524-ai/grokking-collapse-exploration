"""
Transplant atlas module.

This module orchestrates evaluating a grid of component transplants from a donor model
to a recipient model. It leverages the importance scoring to transplant the top-k heads
and MLP blocks, optionally fine-tuning and recording test accuracy, loss, and norm drift.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import json
import itertools
from dataclasses import asdict

try:
    from .importance import score_components, ComponentImportance
    from .transplant_rescue import patch_state_dict, rescue_train, evaluate_model, build_model, load_run
    from ..data import DatasetConfig, generate_modular_arithmetic
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transplant.importance import score_components, ComponentImportance
    from transplant.transplant_rescue import patch_state_dict, rescue_train, evaluate_model, build_model, load_run
    from data import DatasetConfig, generate_modular_arithmetic

def get_keys_for_component(comp_name: str, model: nn.Module) -> List[str]:
    """Helper to map our abstract component names to state dict keys."""
    # This handles names like L0_H0 or L0_MLP
    layer_idx = int(comp_name.split('_')[0][1:])
    keys = []
    if 'H' in comp_name:
        # We don't transplant individual heads' weights in patch_state_dict currently,
        # patch_state_dict replaces entire self_attn_in_proj.
        # But wait, patch_state_dict requires 'component' name mapping.
        # In transplant_rescue, we have 'self_attn_in_proj' etc.
        # Here we might need a more granular approach or map head to the whole attention layer for simplicity.
        pass

    return keys

def build_component_state_dict_patch(
    donor_sd: Dict[str, torch.Tensor],
    recipient_sd: Dict[str, torch.Tensor],
    comp_name: str,
    n_heads: int,
    d_model: int
) -> Dict[str, torch.Tensor]:
    """
    Creates a patched state dict where ONLY the specified component is transplanted from donor to recipient.
    """
    patched_sd = {k: v.clone() for k, v in recipient_sd.items()}
    layer_idx = int(comp_name.split('_')[0][1:])

    if 'H' in comp_name:
        head_idx = int(comp_name.split('_')[1][1:])
        d_head = d_model // n_heads

        # Patch in_proj
        for k in patched_sd:
            if f'layers.{layer_idx}.self_attn.in_proj_weight' in k:
                # Shape is (3*d_model, d_model)
                in_proj_d = donor_sd[k].clone()

                patched_sd[k][head_idx*d_head:(head_idx+1)*d_head, :] = in_proj_d[head_idx*d_head:(head_idx+1)*d_head, :]
                patched_sd[k][d_model + head_idx*d_head:d_model + (head_idx+1)*d_head, :] = in_proj_d[d_model + head_idx*d_head:d_model + (head_idx+1)*d_head, :]
                patched_sd[k][2*d_model + head_idx*d_head:2*d_model + (head_idx+1)*d_head, :] = in_proj_d[2*d_model + head_idx*d_head:2*d_model + (head_idx+1)*d_head, :]

            elif f'layers.{layer_idx}.self_attn.in_proj_bias' in k:
                in_proj_d = donor_sd[k].clone()

                patched_sd[k][head_idx*d_head:(head_idx+1)*d_head] = in_proj_d[head_idx*d_head:(head_idx+1)*d_head]
                patched_sd[k][d_model + head_idx*d_head:d_model + (head_idx+1)*d_head] = in_proj_d[d_model + head_idx*d_head:d_model + (head_idx+1)*d_head]
                patched_sd[k][2*d_model + head_idx*d_head:2*d_model + (head_idx+1)*d_head] = in_proj_d[2*d_model + head_idx*d_head:2*d_model + (head_idx+1)*d_head]

            elif f'layers.{layer_idx}.self_attn.out_proj.weight' in k:
                # Shape is (d_model, d_model). Heads are on dim 1.
                out_proj_d = donor_sd[k].clone()
                patched_sd[k][:, head_idx*d_head:(head_idx+1)*d_head] = out_proj_d[:, head_idx*d_head:(head_idx+1)*d_head]

    elif 'MLP' in comp_name:
        for k in patched_sd:
            if f'layers.{layer_idx}.linear1' in k or f'layers.{layer_idx}.linear2' in k:
                patched_sd[k] = donor_sd[k].clone()

    return patched_sd


def run_atlas(
    donor_run_dir: Path,
    recipient_run_dir: Path,
    donor_step: Optional[int] = None,
    recipient_step: Optional[int] = None,
    rescue_steps: int = 0,
    top_k_heads: int = 2,
    output_path: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
):
    """
    Runs the transplant atlas for a given donor and recipient.
    """
    # 1. Load models
    donor_cfg, donor_sd = load_run(donor_run_dir, donor_step)
    recip_cfg, recip_sd = load_run(recipient_run_dir, recipient_step)

    # 2. Build model objects and dataloaders
    donor_model = build_model(donor_cfg, device)
    donor_model.load_state_dict(donor_sd)

    recip_model = build_model(recip_cfg, device)
    recip_model.load_state_dict(recip_sd)

    # Generate data
    dataset_cfg = DatasetConfig(**donor_cfg["dataset"])
    # We use recipient's noise level for training if rescuing
    if rescue_steps > 0:
        dataset_cfg.label_noise = recip_cfg["dataset"].get("label_noise", 0.0)

    train_x, train_y, test_x, test_y = generate_modular_arithmetic(dataset_cfg)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    # 3. Score donor components on clean test data
    scores = score_components(donor_model, (test_x, test_y), (train_x, train_y))  # Using train as corrupted proxy

    # Filter top k heads and all MLPs
    heads = [s for s in scores if s.component_type == 'head']
    mlps = [s for s in scores if s.component_type == 'mlp']

    top_components = heads[:top_k_heads] + mlps

    results = []

    for comp in top_components:
        # Patch
        patched_sd = build_component_state_dict_patch(
            donor_sd, recip_sd, comp.component_name, recip_model.n_heads, recip_model.d_model
        )
        patched_model = build_model(recip_cfg, device)
        patched_model.load_state_dict(patched_sd)

        # Eval
        loss, acc = evaluate_model(patched_model, test_x, test_y, nn.CrossEntropyLoss())

        # Rescue
        rescue_loss, rescue_acc = None, None
        if rescue_steps > 0:
            # Not fully implementing freeze logic here for individual heads, this is a proxy
            patched_model = rescue_train(
                patched_model, train_x, train_y,
                rescue_steps, recip_cfg.get("optimizer", {}),
                component_to_freeze=None
            )
            rescue_loss, rescue_acc = evaluate_model(patched_model, test_x, test_y, nn.CrossEntropyLoss())

        res = {
            "component": comp.component_name,
            "donor_importance": comp.attribution_score,
            "zero_shot_test_acc": acc,
            "zero_shot_test_loss": loss,
            "rescue_test_acc": rescue_acc,
            "rescue_test_loss": rescue_loss,
            "weight_norm": patched_model.get_weight_norm(),
        }
        results.append(res)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Using JSONL for the artifact
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # Also write a pivot table summary (markdown)
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            if not df.empty:
                pivot_cols = ["component", "donor_importance", "zero_shot_test_acc"]
                if rescue_steps > 0:
                    pivot_cols.append("rescue_test_acc")
                summary = df[pivot_cols].sort_values("donor_importance", ascending=False).to_markdown(index=False)
                with open(output_path.with_suffix(".md"), "w") as f:
                    f.write("# Transplant Atlas Summary\n\n")
                    f.write(summary)
        except ImportError:
            pass

    return results

if __name__ == "__main__":
    print("Testing atlas...")
