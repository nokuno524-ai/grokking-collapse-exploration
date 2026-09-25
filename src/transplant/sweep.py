"""
Systematic circuit transplant sweep.
"""
import copy
import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader

from src.model import ModularArithmeticTransformer
from src.transplant.patching import patch_state_dict, patch_head_state_dict, DEFAULT_PATCH_COMPONENTS


@contextmanager
def graft_context(model: torch.nn.Module, donor_sd: Dict[str, torch.Tensor], components: List[str], randomize: bool = False, rng: Optional[torch.Generator] = None):
    """
    Context manager to safely swap components into `model` from `donor_sd`,
    and restore the original state dict on exit.

    If `randomize` is True, a random orthonormal basis is applied instead
    of copying donor weights.
    """
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        current_sd = model.state_dict()
        for comp in components:
            current_sd = patch_state_dict(current_sd, donor_sd, comp, randomize=randomize, rng=rng)
        model.load_state_dict(current_sd)
        yield
    finally:
        model.load_state_dict(original_sd)


def run_sweep(
    model: ModularArithmeticTransformer,
    donor_sds: List[Dict[str, torch.Tensor]],
    components: List[str],
    eval_fn,
    controls: List[str] = ["donor", "sham", "random"],
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """
    Sweep across transplanting various components into the model, and record
    evaluation metrics.

    Args:
        model: Recipient model.
        donor_sds: List of donor state dicts to transplant from (for multiple seeds).
        components: List of components to try transplanting.
        eval_fn: Function `eval_fn(model)` that returns a dict of metrics.
        controls: List of control types to run ('donor', 'sham', 'random').

    Returns:
        Dictionary of results.
    """
    if rng is None:
        rng = torch.Generator().manual_seed(42)

    results = {}

    # Baseline
    model.eval()
    baseline_metrics = eval_fn(model)
    results["baseline"] = baseline_metrics

    # Sweep over controls, components, and donors
    for comp in components:
        results[comp] = {}
        for control in controls:
            results[comp][control] = []

            if control == "sham":
                # Self-graft
                with graft_context(model, model.state_dict(), [comp], randomize=False):
                    metrics = eval_fn(model)
                    results[comp][control].append(metrics)

            elif control == "random":
                # Random basis swap
                with graft_context(model, {}, [comp], randomize=True, rng=rng):
                    metrics = eval_fn(model)
                    results[comp][control].append(metrics)

            elif control == "donor":
                for donor_sd in donor_sds:
                    with graft_context(model, donor_sd, [comp], randomize=False):
                        metrics = eval_fn(model)
                        results[comp][control].append(metrics)

    return results

def aggregate_sweep_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes mean and bootstrap confidence intervals for the metrics.
    """
    aggregated = {}
    aggregated["baseline"] = results["baseline"]

    for comp, controls in results.items():
        if comp == "baseline":
            continue
        aggregated[comp] = {}
        for control, metrics_list in controls.items():
            if not metrics_list:
                continue

            # Assuming metrics are float values
            keys = metrics_list[0].keys()
            agg_metrics = {}
            for k in keys:
                vals = [m[k] for m in metrics_list if k in m]
                if vals:
                    agg_metrics[f"{k}_mean"] = float(np.mean(vals))
                    # Basic 95% CI using standard error (simplified for demo)
                    stderr = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    agg_metrics[f"{k}_ci"] = 1.96 * stderr
            aggregated[comp][control] = agg_metrics

    return aggregated

def write_sweep_report(aggregated_results: Dict[str, Any], output_path: Path):
    """
    Write sweep results to a Markdown table and JSON.
    """
    # JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    # Markdown
    lines = []
    lines.append("# Transplant Robustness Sweep\n\n")

    baseline = aggregated_results.get("baseline", {})
    lines.append("## Baseline Metrics\n")
    for k, v in baseline.items():
        lines.append(f"- **{k}**: {v:.4f}\n")
    lines.append("\n")

    lines.append("## Interventions\n\n")
    lines.append("| Component | Control | Test Acc (Mean) | Test Acc (CI) | Other Metrics |\n")
    lines.append("|---|---|---|---|---|\n")

    for comp, controls in aggregated_results.items():
        if comp == "baseline":
            continue
        for control, metrics in controls.items():
            acc_mean = metrics.get("test_acc_mean", 0.0)
            acc_ci = metrics.get("test_acc_ci", 0.0)

            other = []
            for k, v in metrics.items():
                if "test_acc" not in k:
                    other.append(f"{k}={v:.3f}")
            other_str = ", ".join(other)

            lines.append(f"| {comp} | {control} | {acc_mean:.4f} | ±{acc_ci:.4f} | {other_str} |\n")

    with open(output_path, "w") as f:
        f.write("".join(lines))
