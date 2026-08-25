"""
Generates experiment configurations and run scripts for a parameter sweep.
Takes a base JSON configuration and a sweep specification (JSON or YAML),
and creates one directory per run with a deterministic name.
"""

import argparse
import json
import itertools
import os
import csv
from pathlib import Path
import copy
import sys

def parse_args():
    """Parse command-line arguments for sweep generation."""
    parser = argparse.ArgumentParser(description="Generate sweep directories for experiments.")
    parser.add_argument("--base", required=True, help="Base JSON config file.")
    parser.add_argument("--spec", required=True, help="YAML or JSON sweep specification file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for runs.")
    return parser.parse_args()

def load_json_or_yaml(path):
    """
    Load a file as either JSON or YAML based on its extension.
    Requires pyyaml to be installed if a YAML file is provided.
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)

def generate_runs():
    """
    Main execution logic to read the base config and sweep specification,
    compute the Cartesian product of all parameters and seeds,
    and generate deterministic run directories with scripts and a manifest.
    """
    args = parse_args()

    base_config = load_json_or_yaml(args.base)
    spec = load_json_or_yaml(args.spec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract name and seeds
    exp_name = spec.get("name", "sweep")
    seeds = spec.get("seeds", [42])
    params = spec.get("parameters", {})

    # Ensure parameter lists
    for k, v in params.items():
        if not isinstance(v, list):
            params[k] = [v]

    # Include seeds in the cartesian product
    param_keys = list(params.keys())
    param_values = [params[k] for k in param_keys]

    manifest_path = out_dir / "manifest.csv"
    manifest_exists = manifest_path.exists()

    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if new
        if not manifest_exists:
            header = ["run_dir", "exp_name"] + param_keys + ["seed"]
            writer.writerow(header)

        # Cartesian product of parameter values and seeds
        for values in itertools.product(*param_values):
            for seed in seeds:
                # Merge config
                run_config = copy.deepcopy(base_config)

                # Assign params
                param_str_parts = []
                for k, v in zip(param_keys, values):
                    run_config[k] = v
                    param_str_parts.append(f"{k}={v}")

                run_config["seed"] = seed

                # Directory name
                param_str = "_".join(param_str_parts) if param_str_parts else "base"
                run_name = f"{exp_name}_{param_str}_seed{seed}"

                run_dir = out_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                # Update output_dir and condition_name for train.py
                run_config["output_dir"] = str(run_dir)
                run_config["condition_name"] = run_name

                # Write config.json
                with open(run_dir / "config.json", "w") as jf:
                    json.dump(run_config, jf, indent=2)

                # Write run.sh
                run_script_path = run_dir / "run.sh"
                with open(run_script_path, "w") as rs:
                    rs.write("#!/bin/bash\n")
                    rs.write(f"# Auto-generated run script for {run_name}\n\n")
                    rs.write(f"python -c '\n")
                    rs.write(f"from src.train import TrainConfig, train\n")
                    rs.write(f"import json\n")
                    rs.write(f"with open(\"config.json\", \"r\") as f:\n")
                    rs.write(f"    c = json.load(f)\n")
                    rs.write(f"config = TrainConfig(**c)\n")
                    rs.write(f"train(config)\n")
                    rs.write(f"'\n")

                os.chmod(run_script_path, 0o755)

                # Manifest entry
                row = [str(run_dir), exp_name] + list(values) + [seed]
                writer.writerow(row)

if __name__ == "__main__":
    generate_runs()
