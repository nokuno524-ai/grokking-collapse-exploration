import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.model import ModularArithmeticTransformer
    from src.data import generate_modular_arithmetic, DatasetConfig
    from src.train import evaluate
    from src.averaging.weight_averaging import load_checkpoint, interpolate_weights, average_weights
except ImportError:
    from model import ModularArithmeticTransformer
    from data import generate_modular_arithmetic, DatasetConfig
    from train import evaluate
    from averaging.weight_averaging import load_checkpoint, interpolate_weights, average_weights


def get_dataloader(config: DatasetConfig, batch_size: int = 512) -> DataLoader:
    _, _, test_inputs, test_targets = generate_modular_arithmetic(config)
    test_dataset = TensorDataset(test_inputs, test_targets)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def run_interpolation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    sd_pre: dict,
    sd_post: dict,
    alphas: list,
    device: torch.device
) -> list:
    results = []
    for alpha in alphas:
        print(f"  Interpolating alpha = {alpha:.2f}")
        interp_sd = interpolate_weights(sd_pre, sd_post, alpha)
        model.load_state_dict(interp_sd, strict=True)
        model.to(device)

        loss, acc = evaluate(model, dataloader, device)
        results.append({
            "alpha": alpha,
            "loss": loss,
            "acc": acc
        })
    return results

def run_swa(
    model: torch.nn.Module,
    dataloader: DataLoader,
    checkpoints: list,
    window_sizes: list,
    device: torch.device
) -> list:
    results = []
    # checkpoints is a list of (step, state_dict) sorted by step

    for window in window_sizes:
        if window > len(checkpoints):
            continue

        # we average the last 'window' checkpoints
        sds = [ckpt for step, ckpt in checkpoints[-window:]]

        print(f"  SWA window = {window}")
        avg_sd = average_weights(sds)
        model.load_state_dict(avg_sd, strict=True)
        model.to(device)

        loss, acc = evaluate(model, dataloader, device)
        results.append({
            "window_size": window,
            "loss": loss,
            "acc": acc
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub-run", action="store_true", help="Run a fast stub evaluation")
    parser.add_argument("--conditions", nargs="+", default=["pure", "low_collapse", "medium_collapse", "severe_collapse"])
    parser.add_argument("--base-dir", type=str, default="results")
    parser.add_argument("--out-path", type=str, default="analysis/averaging_results.json")
    parser.add_argument("--pre-step", type=int, default=5000, help="Early checkpoint step for interpolation")
    parser.add_argument("--post-step", type=int, default=50000, help="Late checkpoint step for interpolation")
    parser.add_argument("--num-alphas", type=int, default=11)
    args = parser.parse_args()

    device = torch.device("cpu")

    # alphas: e.g. 11 points from 0.0 to 1.0
    alphas = np.linspace(0.0, 1.0, args.num_alphas).tolist()

    all_results = {}

    for condition in args.conditions:
        print(f"Processing condition: {condition}")
        cond_dir = Path(args.base_dir) / condition
        if not cond_dir.exists():
            print(f"  Directory {cond_dir} not found. Skipping.")
            continue

        # load config if exists
        results_json_path = cond_dir / "results.json"
        config_kwargs = {}
        if results_json_path.exists():
            with open(results_json_path, "r") as f:
                res_data = json.load(f)
                if "config" in res_data:
                    c = res_data["config"]
                    # extract relevant keys for dataset
                    if "collapse_severity" in c: config_kwargs["collapse_severity"] = c["collapse_severity"]
                    if "collapse_level" in c: config_kwargs["collapse_level"] = c["collapse_level"]

        data_config = DatasetConfig(**config_kwargs)
        if args.stub_run:
            data_config.prime = 7  # much smaller for fast run

        dataloader = get_dataloader(data_config)
        model = ModularArithmeticTransformer(prime=data_config.prime)

        # Load checkpoints for SWA and interpolation
        checkpoint_files = list(cond_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            print(f"  No checkpoints found in {cond_dir}. Skipping.")
            continue

        checkpoints_dict = {}
        for f in checkpoint_files:
            try:
                # e.g. checkpoint_5000.pt -> 5000
                step = int(f.stem.split("_")[1])
                checkpoints_dict[step] = f
            except ValueError:
                continue

        sorted_steps = sorted(checkpoints_dict.keys())

        if args.pre_step not in checkpoints_dict or args.post_step not in checkpoints_dict:
            print(f"  Missing pre/post step ({args.pre_step}, {args.post_step}) in {cond_dir}. Skipping interpolation.")
            interp_results = []
        else:
            sd_pre = load_checkpoint(str(checkpoints_dict[args.pre_step]), device=device)
            sd_post = load_checkpoint(str(checkpoints_dict[args.post_step]), device=device)

            # handle stub run (we need state dicts that match the smaller model)
            if args.stub_run:
                # Mock state dicts instead of using real ones
                sd_pre = model.state_dict()
                sd_post = model.state_dict()

            interp_results = run_interpolation(model, dataloader, sd_pre, sd_post, alphas, device)

        # For SWA, just load all we can
        # To avoid loading too many in memory, we just load them when needed or hold them
        checkpoints_data = []
        if args.stub_run:
            checkpoints_data = [(i, model.state_dict()) for i in range(5)]
        else:
            for step in sorted_steps:
                checkpoints_data.append((step, load_checkpoint(str(checkpoints_dict[step]), device=device)))

        swa_windows = [2, 3, 4, 5]
        if args.stub_run:
            swa_windows = [2, 3]

        swa_results = run_swa(model, dataloader, checkpoints_data, swa_windows, device)

        all_results[condition] = {
            "interpolation": interp_results,
            "swa": swa_results
        }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out_path}")

if __name__ == "__main__":
    main()
