import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from contextlib import contextmanager
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

@contextmanager
def graft_context(model: nn.Module):
    """Context manager for safe in-place state dict cloning and restoration."""
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        yield
    finally:
        model.load_state_dict(original_sd)

from src.transplant.transplant_rescue import (
    load_run,
    build_model,
    make_loaders,
    evaluate_model,
    patch_state_dict,
    COMPONENT_PATTERNS
)
from src.transplant.head_transplant import transplant_attention_head

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-seed-dir", type=str, default="results/multi_seed")
    parser.add_argument("--output-dir", type=str, default="analysis/transplant_sweep")
    args = parser.parse_args()

    multi_seed_dir = Path(args.multi_seed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    severities = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]
    seeds = ["42", "43", "44", "45", "46"]

    # Matched training steps: pre-grokking (5000), grokking transition (15000), post-grokking (50000)
    steps_to_test = [5000, 15000, 50000]

    components_to_test = ["token_embed", "self_attn_in_proj", "self_attn_out_proj", "linear1", "linear2", "output_head"]

    # Data structure: matrix[step][direction][severity][component] = [test_acc1, test_acc2, ...]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    def run_pair(donor_run, base_run, direction, severity, step):
        if not donor_run.exists() or not base_run.exists():
            return

        print(f"Running {direction} for {severity} at step {step} (donor: {donor_run}, base: {base_run})")

        try:
            donor_sd, donor_cfg = load_run(donor_run, step=step)
            base_sd, base_cfg = load_run(base_run, step=step)

            model = build_model(base_cfg, device)
            train_loader, test_loader = make_loaders(base_cfg, device=device)

            # Baseline base model
            model.load_state_dict(base_sd, strict=False)
            baseline_metrics = evaluate_model(model, train_loader, test_loader, device)
            results[step][direction][severity]["baseline_base"].append(baseline_metrics["test_acc"])

            # Baseline donor model evaluated on base data (if seeds match, this is valid)
            model.load_state_dict(donor_sd, strict=False)
            baseline_donor_metrics = evaluate_model(model, train_loader, test_loader, device)
            results[step][direction][severity]["baseline_donor"].append(baseline_donor_metrics["test_acc"])

            for comp in components_to_test:
                patched_sd = patch_state_dict(base_sd, donor_sd, component=comp)
                model.load_state_dict(patched_sd, strict=False)
                patched_metrics = evaluate_model(model, train_loader, test_loader, device)
                results[step][direction][severity][comp].append(patched_metrics["test_acc"])

            # Feature attribution: specific attention heads vs MLPs
            # MLP layers are already tested (linear1, linear2)
            # Test individual attention heads (assuming 4 heads based on defaults)
            d_model = int(base_cfg.get("d_model", 128))
            n_heads = int(base_cfg.get("n_heads", 4))
            for head_idx in range(n_heads):
                patched_sd = transplant_attention_head(
                    base_sd, donor_sd, layer_idx=0, head_idx=head_idx, d_model=d_model, n_heads=n_heads
                )
                model.load_state_dict(patched_sd, strict=False)
                patched_metrics = evaluate_model(model, train_loader, test_loader, device)
                results[step][direction][severity][f"head_{head_idx}"].append(patched_metrics["test_acc"])

        except Exception as e:
            print(f"Failed to process pair: {e}")

    for step in steps_to_test:
        for seed in seeds:
            pure_run = multi_seed_dir / seed / "pure"

            for severity in severities:
                if severity == "pure":
                    continue

                collapse_run = multi_seed_dir / seed / severity

                # pure -> collapsed
                run_pair(donor_run=pure_run, base_run=collapse_run, direction="pure->collapsed", severity=severity, step=step)

                # collapsed -> pure
                run_pair(donor_run=collapse_run, base_run=pure_run, direction="collapsed->pure", severity=severity, step=step)

    # Compute stats
    stats = {}
    for step, step_results in results.items():
        stats[step] = {}
        for direction, direction_results in step_results.items():
            stats[step][direction] = {}
            for severity, sev_results in direction_results.items():
                stats[step][direction][severity] = {}
                for comp, accs in sev_results.items():
                    if len(accs) > 0:
                        stats[step][direction][severity][comp] = {
                            "mean": float(np.mean(accs)),
                            "std": float(np.std(accs)),
                            "n": len(accs)
                        }

    with open(out_dir / "transplant_matrix.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Write Markdown
    with open(out_dir / "transplant_matrix.md", "w") as f:
        f.write("# Transplant Success Matrix (Feature Attribution)\n\n")
        f.write("Values are Mean ± Std Test Accuracy over seeds.\n\n")

        for step, step_results in stats.items():
            f.write(f"## Training Step: {step}\n\n")
            for direction, direction_results in step_results.items():
                f.write(f"### Direction: {direction}\n\n")
                f.write("| Component | " + " | ".join([s for s in severities if s != "pure"]) + " |\n")
                f.write("|---|" + "|".join(["---" for _ in severities if _ != "pure"]) + "|\n")

                # Use the components from the first severity found
                first_sev = next(iter(direction_results.keys()))
                components = list(direction_results[first_sev].keys())

                for comp in components:
                    row = [comp]
                    for sev in severities:
                        if sev == "pure": continue
                        if sev in direction_results and comp in direction_results[sev]:
                            mean = direction_results[sev][comp]["mean"]
                            std = direction_results[sev][comp]["std"]
                            row.append(f"{mean:.3f} ± {std:.3f}")
                        else:
                            row.append("N/A")
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n")

if __name__ == "__main__":
    main()
