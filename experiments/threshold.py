import os
import yaml
import json
import argparse
from pathlib import Path
from src.train import train, TrainConfig

def binary_search_threshold(config, seed: int, output_dir: Path):
    """Run binary search to find the collapse threshold where grokking stops."""
    min_c = config.get("min_collapse", 0.0)
    max_c = config.get("max_collapse", 0.5)
    tol = config.get("tolerance", 0.02)
    prime = config.get("prime", 59)
    max_steps = config.get("max_steps", 50000)
    eval_every = config.get("eval_every", 100)

    print(f"\\n{'='*60}")
    print(f"Starting binary search for seed {seed}")
    print(f"{'='*60}")

    results = []

    while (max_c - min_c) > tol:
        mid_c = (min_c + max_c) / 2
        condition_name = f"search_seed{seed}_col{mid_c:.4f}"
        print(f"Testing collapse_level = {mid_c:.4f} (range: {min_c:.4f} - {max_c:.4f})")

        train_config = TrainConfig(
            prime=prime,
            collapse_level=mid_c,
            max_steps=max_steps,
            eval_every=eval_every,
            seed=seed,
            condition_name=condition_name,
            output_dir=str(output_dir)
        )

        state = train(train_config)

        results.append({
            "collapse_level": mid_c,
            "grokked": state.grokked,
            "grokking_step": state.grokking_step,
            "final_test_acc": state.test_acc
        })

        if state.grokked:
            # Model grokked, so we can probably push collapse higher
            min_c = mid_c
        else:
            # Model didn't grok, collapse was too high
            max_c = mid_c

    # The threshold is approximately min_c (the highest collapse where it grokked)
    threshold = min_c
    print(f"\\nFound threshold for seed {seed}: ~{threshold:.4f}")
    return threshold, results

def run_threshold_experiment(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [42, 43, 44, 45, 46])
    output_dir = Path(config.get("output_dir", "results/threshold"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_thresholds = []
    all_search_history = []

    for seed in seeds:
        threshold, history = binary_search_threshold(config, seed, output_dir)
        all_thresholds.append({"seed": seed, "threshold": threshold})
        all_search_history.extend(history)

    with open(output_dir / "thresholds_summary.json", 'w') as f:
        json.dump({"thresholds": all_thresholds, "history": all_search_history}, f, indent=2)

    print(f"\\nThreshold experiments complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/threshold.yaml")
    args = parser.parse_args()
    run_threshold_experiment(args.config)
