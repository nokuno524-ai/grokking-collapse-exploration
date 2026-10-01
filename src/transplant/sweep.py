"""
Systematic circuit transplant sweep between checkpoints from different seeds/runs.
Outputs a localization matrix of test accuracy changes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# We assume standard project layout for these imports
from .patching import patch_state_dict, patch_head_state_dict
from src.transplant_rescue import COMPONENT_PATTERNS
from src.rigorous_stats.stats_module import format_stat_result

def induction_train(
    model: nn.Module,
    component_name: str,
    train_loader: DataLoader,
    device: torch.device,
    steps: int = 100,
    lr: float = 1e-3
) -> None:
    """
    Induction probing: freeze all parameters except the targeted transplanted component,
    and briefly fine-tune the component on the recipient's dataset.
    """
    model.train()
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the specific component
    import re
    if component_name in COMPONENT_PATTERNS:
        pat = COMPONENT_PATTERNS[component_name]
        for name, param in model.named_parameters():
            if re.match(pat, name):
                param.requires_grad = True
    else:
        # Fallback if it's a specific param name
        for name, param in model.named_parameters():
            if name == component_name:
                param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    steps_done = 0
    while steps_done < steps:
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            steps_done += 1
            if steps_done >= steps:
                break

def evaluate_batches(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> List[float]:
    """
    Evaluate the model, returning the accuracy for each batch in the test loader.
    Useful for calculating effect size distributions.
    """
    model.eval()
    batch_accs = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            batch_accs.append(acc)
    return batch_accs

def run_transplant_sweep(
    donor_sd: Dict[str, torch.Tensor],
    recipient_sd: Dict[str, torch.Tensor],
    recipient_model: nn.Module,
    recipient_test_loader: DataLoader,
    components_to_sweep: List[str],
    device: torch.device,
    key_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Systematically swap components from donor to recipient and measure task-performance retention.
    Returns a dictionary mapping component name to summary metrics.
    """
    results = {}

    # Baseline
    recipient_model.load_state_dict(recipient_sd)
    baseline_accs = evaluate_batches(recipient_model, recipient_test_loader, device)
    import numpy as np
    baseline_mean = float(np.mean(baseline_accs))
    logging.info(format_stat_result(baseline_accs, "Baseline Recipient Accuracy"))

    for comp in components_to_sweep:
        try:
            # Try module level patch first
            patched_sd = patch_state_dict(recipient_sd, donor_sd, comp, key_mapping=key_mapping)
            recipient_model.load_state_dict(patched_sd, strict=False)

            accs = evaluate_batches(recipient_model, recipient_test_loader, device)
            mean_acc = float(np.mean(accs))
            delta = mean_acc - baseline_mean

            results[comp] = {
                "mean_acc": mean_acc,
                "delta": delta,
                "batch_accs": accs # raw distribution for CI later
            }
            logging.info(format_stat_result(accs, f"Transplant {comp} Accuracy"))

        except Exception as e:
            logging.error(f"Failed to transplant {comp}: {e}")
            results[comp] = {"error": str(e)}

    return results

def plot_transplant_matrix(
    results_matrix: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path
) -> None:
    """
    Plots a heatmap from a nested dictionary: {donor_seed: {component: {delta: float, ...}}}
    """
    import numpy as np

    donor_seeds = list(results_matrix.keys())
    if not donor_seeds:
        return

    components = [k for k, v in results_matrix[donor_seeds[0]].items() if "error" not in v]

    matrix = np.zeros((len(donor_seeds), len(components)))
    for i, seed in enumerate(donor_seeds):
        for j, comp in enumerate(components):
            matrix[i, j] = results_matrix[seed][comp].get("delta", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(components) * 0.5), max(5, len(donor_seeds) * 0.5)))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(donor_seeds)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticklabels(donor_seeds)
    ax.set_xlabel("Component Transplanted")
    ax.set_ylabel("Donor Seed")
    ax.set_title("Cross-Run Transplant Success (Accuracy Delta)")

    fig.colorbar(im, ax=ax, label="Test Accuracy Change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_transplant_matrix(
    results_matrix: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path
) -> None:
    """Save the transplant results matrix as a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results_matrix, f, indent=2)

def check_seed_leakage(donor_cfg: dict, recipient_cfg: dict) -> None:
    """
    Explicitly guard against seed leakage between donor and recipient checkpoints.
    If they have the same seed, they have the same train/test split.
    """
    donor_seed = donor_cfg.get("seed", -1)
    recipient_seed = recipient_cfg.get("seed", -2)
    if donor_seed == recipient_seed:
        raise ValueError(
            f"Seed leakage detected! Donor seed {donor_seed} matches "
            f"recipient seed {recipient_seed}. Cross-run transplant invalid."
        )
