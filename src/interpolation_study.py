import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.train import train, TrainConfig

def run_interpolation_study(out_dir: str = "results/interpolation", max_steps: int = 50000):
    """
    Run an interpolation study mixing clean and collapsed data at various ratios
    to find the critical threshold for grokking prevention.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We will vary the collapse_level (fraction of collapsed data) from 0.0 to 0.5
    # with a fixed severe collapse generator
    levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    results = {}

    for lvl in levels:
        print(f"\n--- Running interpolation level: {lvl} ---")
        condition_name = f"interp_{lvl}"

        config = TrainConfig(
            collapse_level=lvl,
            collapse_severity=0.7, # High collapse
            condition_name=condition_name,
            output_dir=str(out_dir),
            max_steps=max_steps,
            eval_every=500, # More frequent eval to see fine-grained transition
            log_every=500
        )

        state = train(config)

        results[lvl] = {
            "grokked": state.grokked,
            "grokking_step": state.grokking_step,
            "final_test_acc": state.test_acc,
            "final_train_acc": state.train_acc,
        }

    # Save aggregate results
    res_path = out_dir / "interpolation_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Interpolation study complete. Results saved to {res_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50000)
    args = parser.parse_args()

    run_interpolation_study(max_steps=args.max_steps)
