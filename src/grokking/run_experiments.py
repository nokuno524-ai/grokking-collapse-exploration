import os
import sys
import yaml
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pydantic import BaseModel, Field, conlist, model_validator
from typing import List, Optional

class ExperimentRunConfig(BaseModel):
    model_size: dict = Field(..., description="Model architecture parameters like d_model, n_heads, n_layers")
    dataset: dict = Field(..., description="Dataset parameters like prime, train_fraction")
    composition_ratios: List[float] = Field(..., description="List of dataset composition ratios (real vs synthetic), maps directly to collapse_level")
    collapse_severities: List[float] = Field(..., description="List of collapse severity levels")
    training_steps: int = Field(..., gt=0, description="Number of training steps")
    seeds: List[int] = Field(..., description="List of random seeds")
    output_dir: str = Field("results", description="Root output directory")

    @model_validator(mode='after')
    def validate_ratios(self) -> 'ExperimentRunConfig':
        for ratio in self.composition_ratios:
            if not (0.0 <= ratio <= 1.0):
                raise ValueError("composition_ratios must be between 0.0 and 1.0")
        for sev in self.collapse_severities:
            if not (0.0 <= sev <= 1.0):
                raise ValueError("collapse_severities must be between 0.0 and 1.0")
        return self

def load_and_validate_config(yaml_path: str) -> ExperimentRunConfig:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return ExperimentRunConfig(**data)

def run_single_experiment(kwargs: dict) -> bool:
    """Run a single experiment with the given kwargs for TrainConfig."""
    import sys
    import os
    # Add repo root to path to ensure src can be found in subprocesses
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from src.train import train, TrainConfig

        condition_name = kwargs.get("condition_name", "unnamed")
        output_dir = kwargs.get("output_dir", "results")

        target_dir = Path(output_dir) / condition_name
        results_file = target_dir / "results.json"

        if results_file.exists():
            print(f"Skipping {condition_name}, results.json already exists.")
            return True

        print(f"Starting experiment: {condition_name}")
        train_config = TrainConfig(**kwargs)
        train(train_config)
        return True
    except Exception as e:
        print(f"Experiment {kwargs.get('condition_name')} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run grokking experiments from a YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    config = load_and_validate_config(args.config)
    print(f"Loaded config from {args.config}")

    tasks = []
    # Build the grid of tasks
    for ratio in config.composition_ratios:
        for sev in config.collapse_severities:
            for seed in config.seeds:
                condition_name = f"ratio_{ratio}_sev_{sev}_seed_{seed}"

                kwargs = {
                    "condition_name": condition_name,
                    "output_dir": config.output_dir,
                    "max_steps": config.training_steps,
                    "seed": seed,
                    "collapse_level": ratio,
                    "collapse_severity": sev,
                }

                # Update with model sizing and dataset configurations
                kwargs.update(config.model_size)
                kwargs.update(config.dataset)

                tasks.append(kwargs)

    print(f"Total experiments to run: {len(tasks)}")

    if args.parallel:
        print(f"Running in parallel with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_single_experiment, task) for task in tasks]
            for i, future in enumerate(as_completed(futures)):
                success = future.result()
                if not success:
                    print(f"Task {i} failed.")
    else:
        print("Running sequentially")
        for task in tasks:
            success = run_single_experiment(task)
            if not success:
                print(f"Experiment {task['condition_name']} failed.")

if __name__ == "__main__":
    main()
