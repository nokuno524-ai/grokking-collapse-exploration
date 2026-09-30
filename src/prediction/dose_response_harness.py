"""
Dose-response harness for sweeping synthetic-data fraction and mixture temperature.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

# Add src to sys.path so we can import from train
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from train import TrainConfig, train
except ImportError:
    from src.train import TrainConfig, train


def generate_configs(levels, severities, seeds, output_root="results/dose_response"):
    """
    Generate training configurations sweeping across levels and severities.

    Args:
        levels (list of float): Fractions of synthetic data (e.g. collapse_level).
        severities (list of float): Mixture temperatures (e.g. collapse_severity).
        seeds (list of int): Random seeds for reproducibility.
        output_root (str): Base directory for saving output runs.

    Returns:
        list of TrainConfig: The generated configurations.
    """
    configs = []
    for level in levels:
        for severity in severities:
            for seed in seeds:
                pair_dir = f"level{level:g}_sev{severity:g}"
                configs.append(TrainConfig(
                    prime=59,
                    train_fraction=0.3,
                    lr=1e-3,
                    weight_decay=1.0,
                    collapse_level=level,
                    collapse_severity=severity,
                    seed=seed,
                    condition_name=f"seed_{seed}",
                    output_dir=str(Path(output_root) / pair_dir),
                    max_steps=50000,
                ))
    return configs


def main():
    """CLI for the dose-response harness."""
    parser = argparse.ArgumentParser(description="Generate dose-response configs.")
    parser.add_argument("--levels", type=str, default="0.0,0.1,0.2,0.3,0.4",
                        help="Comma-separated list of collapse_level values.")
    parser.add_argument("--severities", type=str, default="0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated list of collapse_severity values.")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46",
                        help="Comma-separated list of random seeds.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate and print configs without running them.")
    parser.add_argument("--output-root", type=str, default="results/dose_response",
                        help="Root directory for saving results.")
    args = parser.parse_args()

    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    severities = [float(x) for x in args.severities.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    configs = generate_configs(levels, severities, seeds, args.output_root)

    if args.dry_run:
        print(f"Generated {len(configs)} configurations (dry-run):")
        for cfg in configs:
            print(json.dumps(asdict(cfg)))
    else:
        for i, cfg in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] Running config: level={cfg.collapse_level}, sev={cfg.collapse_severity}, seed={cfg.seed}")
            train(cfg)


if __name__ == "__main__":
    main()
