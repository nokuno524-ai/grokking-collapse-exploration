import argparse
import os
import sys
from pathlib import Path

# Add src to sys.path to allow importing train
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import TrainConfig, train

def run_scaling_experiment(
    d_model, n_heads, train_fraction, collapse_level, output_root="results/scaling", max_steps=50000, seed=42
):
    """
    Run a scaling experiment with specific hyperparameters.
    """
    # Number of layers is kept to 1 according to the architecture context
    n_layers = 1

    # Calculate d_ff
    d_ff = d_model * 4

    condition_name = f"d{d_model}_h{n_heads}_tf{train_fraction}_cl{collapse_level}_s{seed}"
    output_dir = Path(output_root) / f"d{d_model}_h{n_heads}" / f"tf{train_fraction}_cl{collapse_level}"

    print(f"Running config: d_model={d_model}, n_heads={n_heads}, train_fraction={train_fraction}, collapse_level={collapse_level}, seed={seed}")

    config = TrainConfig(
        prime=59,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        train_fraction=train_fraction,
        collapse_level=collapse_level,
        collapse_severity=0.5, # default
        seed=seed,
        condition_name=condition_name,
        output_dir=str(output_dir),
        max_steps=max_steps
    )

    return train(config)

def get_scaling_matrix():
    """
    Returns lists of parameters to sweep over.
    """
    d_models = [64, 128, 256]
    n_heads = [2, 4, 8]
    train_fractions = [0.2, 0.3, 0.4]
    collapse_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    # Let's filter some out or just return the combination.
    # For a full grid this would be 3*3*3*6 = 162 runs per seed.
    return {
        "d_models": d_models,
        "n_heads": n_heads,
        "train_fractions": train_fractions,
        "collapse_levels": collapse_levels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--collapse-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/scaling")
    parser.add_argument("--run-all", action="store_true")

    args = parser.parse_args()

    if args.run_all:
        matrix = get_scaling_matrix()
        for d in matrix["d_models"]:
            for h in matrix["n_heads"]:
                for tf in matrix["train_fractions"]:
                    for cl in matrix["collapse_levels"]:
                        run_scaling_experiment(d, h, tf, cl, args.output_dir, args.max_steps, args.seed)
    else:
        run_scaling_experiment(args.d_model, args.n_heads, args.train_fraction, args.collapse_level, args.output_dir, args.max_steps, args.seed)

if __name__ == "__main__":
    main()
