import argparse
import itertools
import json
import os
from pathlib import Path
from dataclasses import asdict

from src.train import train, TrainConfig
from src.data import get_all_conditions

from typing import List, Dict, Any, Union

def get_model_sizes() -> List[Dict[str, Union[str, int]]]:
    """Returns a list of model size configurations."""
    return [
        {"name": "tiny", "d_model": 16, "n_heads": 2, "d_ff": 32, "n_layers": 1},
        {"name": "small", "d_model": 64, "n_heads": 4, "d_ff": 128, "n_layers": 1},
        {"name": "base", "d_model": 128, "n_heads": 4, "d_ff": 512, "n_layers": 1}
    ]

def get_data_sizes() -> List[float]:
    """Returns a list of data fractions to train on."""
    return [0.2, 0.4, 0.6, 0.8]

def get_collapse_severities() -> List[str]:
    """Returns a list of condition names representing collapse severities."""
    # Use standard conditions: pure (0.0), medium (0.5), severe (0.9)
    return ["pure", "medium_collapse", "severe_collapse"]

def run_scaling_grid(out_file: str, smoke_test: bool = False) -> None:
    """Run a grid of experiments and append results to a JSONL file."""

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        model_sizes = [{"name": "tiny", "d_model": 16, "n_heads": 2, "d_ff": 32, "n_layers": 1}]
        data_sizes = [0.1]
        severities = ["pure"]
        max_steps = 10
        eval_every = 2
    else:
        model_sizes = get_model_sizes()
        data_sizes = get_data_sizes()
        severities = get_collapse_severities()
        max_steps = 25000
        eval_every = 100

    all_conditions = get_all_conditions()

    for model, data_fraction, severity_name in itertools.product(model_sizes, data_sizes, severities):
        print(f"Running: Model={model['name']}, Data={data_fraction}, Condition={severity_name}")

        condition_cfg = all_conditions[severity_name]

        config = TrainConfig(
            prime=59 if not smoke_test else 11,
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            d_ff=model["d_ff"],
            n_layers=model["n_layers"],
            train_fraction=data_fraction,
            collapse_level=condition_cfg.collapse_level,
            collapse_severity=condition_cfg.collapse_severity,
            condition_name=severity_name,
            output_dir=f"results_scaling/m_{model['name']}_d_{data_fraction}_{severity_name}",
            max_steps=max_steps,
            eval_every=eval_every,
            log_every=eval_every * 5,
        )

        state = train(config)

        result_record = {
            "model_size_name": model["name"],
            "d_model": model["d_model"],
            "n_heads": model["n_heads"],
            "d_ff": model["d_ff"],
            "n_layers": model["n_layers"],
            "train_fraction": data_fraction,
            "condition_name": severity_name,
            "collapse_level": condition_cfg.collapse_level,
            "collapse_severity": condition_cfg.collapse_severity,
            "grokked": state.grokked,
            "grokking_step": state.grokking_step,
            "history": state.history,
        }

        with open(out_path, "a") as f:
            f.write(json.dumps(result_record) + "\n")

    print(f"Done! Results written to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling experiments")
    parser.add_argument("--out-file", type=str, default="results_scaling/scaling_results.jsonl")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    run_scaling_grid(args.out_file, args.smoke_test)
