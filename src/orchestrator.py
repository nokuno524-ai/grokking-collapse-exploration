"""
Orchestrator for grokking-collapse exploration.
Handles running the experimental grid, managing state, validating runs into a central registry,
and generating reports based on the registry safely without running duplicate tasks.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from src.train import TrainConfig, train
    from src.run_grid import DEFAULT_LEVELS, DEFAULT_SEVERITIES, DEFAULT_SEEDS, build_tasks
except ImportError:
    from train import TrainConfig, train
    from run_grid import DEFAULT_LEVELS, DEFAULT_SEVERITIES, DEFAULT_SEEDS, build_tasks


def get_config_hash(config_dict: Dict[str, Any]) -> str:
    """Generate a stable hash for a configuration dictionary."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def update_registry(registry_path: Path, run_results_path: Path) -> None:
    """Read a run's results.json and add/update it in the central registry.json atomically."""
    if not run_results_path.exists():
        return

    try:
        with open(run_results_path, "r") as f:
            run_data = json.load(f)
    except Exception as e:
        print(f"Error reading {run_results_path}: {e}")
        return

    config = run_data.get("config", {})
    if not config:
        return

    config_hash = get_config_hash(config)

    # Extract only what's needed for the registry
    entry = {
        "config": config,
        "metrics": {
            "grokked": run_data.get("grokked", False),
            "grokking_step": run_data.get("grokking_step"),
            "final_test_acc": run_data.get("final_test_acc", 0.0),
            "final_fourier_concentration": run_data.get("final_fourier_concentration", 0.0),
            "final_embedding_rank": run_data.get("final_embedding_rank", 0.0),
        }
    }

    registry_data = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                registry_data = json.load(f)
        except Exception:
            pass

    registry_data[config_hash] = entry

    # Write atomically
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_registry_path = registry_path.with_suffix(".json.tmp")
    with open(tmp_registry_path, "w") as f:
        json.dump(registry_data, f, indent=2)
    os.replace(tmp_registry_path, registry_path)


def get_registry_hash(registry_path: Path) -> str:
    """Get the hash of the current registry.json content to detect staleness."""
    if not registry_path.exists():
        return ""
    with open(registry_path, "r") as f:
        content = f.read()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def cmd_run(args: argparse.Namespace) -> None:
    """Run the training grid and collect results."""
    levels = DEFAULT_LEVELS
    severities = DEFAULT_SEVERITIES
    seeds = DEFAULT_SEEDS
    max_steps = args.max_steps
    output_dir = Path(args.output_dir)

    if args.smoke:
        print("Running in SMOKE mode. Overriding grid to small values.")
        levels = [0.0]
        severities = [0.3]
        seeds = [42]
        max_steps = 2

    tasks = build_tasks(levels, severities, seeds)
    print(f"Running {len(tasks)} tasks...")

    # Load orchestrator state
    state_path = output_dir / "orchestrator_state.json"
    state = {"completed_tasks": []}
    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
        except Exception:
            pass

    registry_path = output_dir / "registry.json"

    for i, (level, severity, seed) in enumerate(tasks):
        task_id = f"level{level:g}_sev{severity:g}_seed{seed}"
        if task_id in state["completed_tasks"]:
            print(f"[{i+1}/{len(tasks)}] Skipping already completed task {task_id}")
            continue

        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] Running {task_id}\n{'='*60}")

        pair_dir = f"level{level:g}_sev{severity:g}"
        out_path = output_dir / pair_dir

        train_config = TrainConfig(
            prime=59 if not args.smoke else 7, # smaller prime for smoke
            train_fraction=0.3,
            lr=1e-3,
            weight_decay=1.0,
            collapse_level=level,
            collapse_severity=severity,
            seed=seed,
            condition_name=f"seed_{seed}",
            output_dir=str(out_path),
            max_steps=max_steps,
        )

        # Execute run
        train(train_config)

        # Update registry
        run_results_path = out_path / f"seed_{seed}" / "results.json"
        update_registry(registry_path, run_results_path)

        # Atomically update state
        state["completed_tasks"].append(task_id)
        tmp_state = state_path.with_suffix(".json.tmp")
        with open(tmp_state, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_state, state_path)

    print("All tasks completed.")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate reports and figures from the registry."""
    registry_path = Path(args.output_dir) / "registry.json"
    if not registry_path.exists():
        print(f"Registry not found at {registry_path}. Run tasks first.")
        sys.exit(1)

    current_hash = get_registry_hash(registry_path)

    # We will simulate generating a report (e.g. figure or summary markdown)
    # and dropping a sidecar JSON with the manifest hash.
    report_dir = Path(args.output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = report_dir / "manifest.json"

    # Check staleness
    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("registry_hash") == current_hash and not args.force:
            print("Reports are up-to-date. Skipping regeneration.")
            return
        else:
            print("Staleness detected: registry has changed. Regenerating reports...")
    else:
        print("No manifest found. Generating reports...")


    # Run analysis scripts
    import subprocess
    import sys

    scripts = [
        "analysis/comprehensive_analysis.py",
        "analysis/contam_analysis.py",
        "analysis/exp_c_grid_analysis.py",
        "analysis/grid_analysis.py",
    ]

    for script in scripts:
        print(f"Running {script}...")
        try:
            # We don't want to crash if one fails, but we should log it
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    # Write manifest atomically

    tmp_sidecar = sidecar_path.with_suffix(".json.tmp")
    with open(tmp_sidecar, "w") as f:
        json.dump({"registry_hash": current_hash}, f, indent=2)
    os.replace(tmp_sidecar, sidecar_path)


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Grokking-collapse orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_run = subparsers.add_parser("run", help="Run the experimental grid")
    parser_run.add_argument("--smoke", action="store_true", help="Run a fast, tiny grid for testing")
    parser_run.add_argument("--max-steps", type=int, default=50000, help="Max training steps")
    parser_run.add_argument("--output-dir", type=str, default="results/grid", help="Output directory")

    parser_report = subparsers.add_parser("report", help="Regenerate reports from the registry")
    parser_report.add_argument("--output-dir", type=str, default="results/grid", help="Output directory containing registry.json")
    parser_report.add_argument("--force", action="store_true", help="Force rebuild even if not stale")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
