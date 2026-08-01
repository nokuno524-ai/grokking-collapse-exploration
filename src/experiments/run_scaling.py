import argparse
from pathlib import Path

# Try importing dependencies cleanly
try:
    from src.train import TrainConfig, train, get_all_conditions
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.train import TrainConfig, train, get_all_conditions

def run_scaling_experiment(output_root: str, max_steps: int, seeds: list[int]):
    """
    Run the 5 collapse conditions over multiple seeds.
    """
    results = {}
    conditions = get_all_conditions() # Gives pure, low_collapse, medium_collapse, etc with default seed 42

    for condition_name, base_config in conditions.items():
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Running condition: {condition_name}, Seed: {seed}")
            print(f"{'='*60}")

            output_dir = str(Path(output_root) / condition_name / f"seed_{seed}")

            train_config = TrainConfig(
                prime=base_config.prime,
                train_fraction=base_config.train_fraction,
                collapse_level=base_config.collapse_level,
                collapse_severity=base_config.collapse_severity,
                noise_fraction=base_config.noise_fraction,
                seed=seed,
                condition_name=condition_name,
                output_dir=output_dir,
                max_steps=max_steps,
            )

            state = train(train_config)

            if condition_name not in results:
                results[condition_name] = []
            results[condition_name].append({
                "seed": seed,
                "grokked": state.grokked,
                "grokking_step": state.grokking_step,
                "final_test_acc": state.test_acc,
                "fourier_concentration": state.fourier_concentration,
            })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/scaling")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_scaling_experiment(args.output_dir, args.max_steps, seeds)
