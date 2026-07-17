import os
import yaml
import json
import argparse
from pathlib import Path
from src.train import train, TrainConfig

def run_scaling_sweep(config_path: str):
    """Run the scaling sweep experiment based on the config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    depths = config.get("depths", [1, 2, 3, 4, 6])
    widths = config.get("widths", [64, 128, 256, 512])
    collapse_levels = config.get("collapse_levels", [0.0, 0.1, 0.3, 0.5, 0.7])
    prime = config.get("prime", 59)
    max_steps = config.get("max_steps", 50000)
    eval_every = config.get("eval_every", 100)
    seeds = config.get("seeds", [42])
    output_dir = Path(config.get("output_dir", "results/scaling"))

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in seeds:
        for depth in depths:
            for width in widths:
                for collapse in collapse_levels:
                    condition_name = f"depth{depth}_width{width}_col{collapse}_seed{seed}"
                    print(f"\\n{'='*60}")
                    print(f"Running condition: {condition_name}")
                    print(f"{'='*60}")

                    train_config = TrainConfig(
                        prime=prime,
                        d_model=width,
                        n_heads=max(1, width // 64), # keep head dim around 64
                        n_layers=depth,
                        collapse_level=collapse,
                        max_steps=max_steps,
                        eval_every=eval_every,
                        seed=seed,
                        condition_name=condition_name,
                        output_dir=str(output_dir)
                    )

                    state = train(train_config)

                    result_summary = {
                        "depth": depth,
                        "width": width,
                        "collapse_level": collapse,
                        "seed": seed,
                        "grokked": state.grokked,
                        "grokking_step": state.grokking_step,
                        "final_test_acc": state.test_acc,
                        "final_train_acc": state.train_acc,
                        "final_weight_norm": state.weight_norm,
                    }
                    all_results.append(result_summary)

    # Save summary of all results
    summary_path = output_dir / "scaling_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\\nScaling sweep complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling sweep experiment")
    parser.add_argument("--config", type=str, default="configs/scaling.yaml", help="Path to config file")
    args = parser.parse_args()

    run_scaling_sweep(args.config)
