import argparse
import yaml
import subprocess
import os
import json
from pathlib import Path
import pandas as pd

def run_experiment(config_path, output_base="results/ablations"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config['name']
    params = config['parameters']
    fixed = config['fixed']

    param_name = list(params.keys())[0]
    param_values = params[param_name]

    results = []

    for val in param_values:
        out_dir = Path(output_base) / exp_name / f"{param_name}_{val}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "src/train.py",
            "--output-dir", str(out_dir)
        ]

        # Add fixed params
        for k, v in fixed.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        # Add sweep param
        if param_name == "noise_fraction":
            cmd.extend(["--collapse-level", str(val)])
        else:
            cmd.extend([f"--{param_name.replace('_', '-')}", str(val)])

        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)

            # Read results
            res_file = out_dir / "results.json"
            if res_file.exists():
                with open(res_file) as rf:
                    data = json.load(rf)
                    results.append({
                        param_name: val,
                        'grokking_step': data.get('grokking_step', 10000),
                        'final_test_acc': data.get('final_test_acc', 0)
                    })
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed: {e}")
            results.append({
                param_name: val,
                'grokking_step': None,
                'final_test_acc': None
            })

    # Save summary
    if results:
        df = pd.DataFrame(results)
        summary_path = Path(output_base) / f"{exp_name}_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to ablation config")
    parser.add_argument("--all", action="store_true", help="Run all configs in configs/ablations")
    args = parser.parse_args()

    if args.all:
        config_dir = Path("configs/ablations")
        for conf in config_dir.glob("*.yaml"):
            run_experiment(conf)
    elif args.config:
        run_experiment(args.config)
    else:
        print("Please provide --config or --all")
