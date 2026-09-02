import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Any

try:
    from src.train import TrainConfig, train
except ImportError:
    from train import TrainConfig, train  # type: ignore

DEFAULT_COLLAPSE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_TRAIN_FRACTIONS = [0.2, 0.3, 0.5, 0.8]
DEFAULT_D_MODELS = [32, 64, 128]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

def fmt_collapse(c: float) -> str:
    return f"c{c:g}"

def fmt_frac(f: float) -> str:
    return f"f{f:g}"

def fmt_dmodel(d: int) -> str:
    return f"d{d}"

def build_tasks(collapse_levels: List[float], train_fractions: List[float], d_models: List[int], seeds: List[int]) -> List[Tuple[float, float, int, int]]:
    """Build the grid of tasks."""
    return [(c, f, d, s) for c in collapse_levels for f in train_fractions for d in d_models for s in seeds]

def run_one(collapse_level: float, train_fraction: float, d_model: int, seed: int, output_root: str, max_steps: int) -> Any:
    out_dir = Path(output_root) / fmt_dmodel(d_model) / fmt_frac(train_fraction) / fmt_collapse(collapse_level)
    condition_name = f"seed_{seed}"
    run_dir = out_dir / condition_name

    results_file = run_dir / "results.json"
    if results_file.exists():
        print(f"Skipping {run_dir} (already exists)")
        return None

    print(f"Running {run_dir}")
    train_config = TrainConfig(
        prime=59,
        train_fraction=train_fraction,
        lr=1e-3,
        weight_decay=1.0,
        d_model=d_model,
        collapse_level=collapse_level,
        collapse_severity=0.5,  # Fixed as per issue logic
        seed=seed,
        condition_name=condition_name,
        output_dir=str(out_dir),
        max_steps=max_steps,
    )
    return train(train_config)

def emit_sbatch(tasks: List[Tuple[float, float, int, int]], output_dir: str, max_steps: int) -> None:
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=phase-diag
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/phasediag-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/phasediag-%A_%a.err
#SBATCH --array=0-{len(tasks) - 1}%50

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX="${{SLURM_ARRAY_TASK_ID:-0}}"
echo "[phasediag] task=${{IDX}} host=$(hostname) gpu=${{CUDA_VISIBLE_DEVICES:-?}}"

python3 -m src.sweep.driver \\
    --array-id "${{IDX}}" \\
    --output-dir {output_dir} \\
    --max-steps {max_steps}

echo "[phasediag] task=${{IDX}} done"
"""
    os.makedirs("slurm", exist_ok=True)
    sbatch_path = "slurm/phase_diagram.sbatch"
    with open(sbatch_path, "w") as f:
        f.write(sbatch_content)
    print(f"Wrote {len(tasks)} tasks to {sbatch_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep harness for scaling phase diagram.")
    parser.add_argument("--collapse-levels", type=str, default=",".join(f"{x:g}" for x in DEFAULT_COLLAPSE_LEVELS))
    parser.add_argument("--train-fractions", type=str, default=",".join(f"{x:g}" for x in DEFAULT_TRAIN_FRACTIONS))
    parser.add_argument("--d-models", type=str, default=",".join(str(x) for x in DEFAULT_D_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/phase_diagram")
    parser.add_argument("--array-id", type=int, default=None, help="Run only the task at this index.")
    parser.add_argument("--emit-sbatch", action="store_true", help="Generate sbatch script and exit.")
    parser.add_argument("--print-task-count", action="store_true", help="Just print the total number of tasks and exit.")

    args = parser.parse_args()

    c_levels = [float(x) for x in args.collapse_levels.split(",") if x]
    t_fracs = [float(x) for x in args.train_fractions.split(",") if x]
    d_mods = [int(x) for x in args.d_models.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]

    tasks = build_tasks(c_levels, t_fracs, d_mods, seeds)

    if args.print_task_count:
        print(len(tasks))
        return

    if args.emit_sbatch:
        emit_sbatch(tasks, args.output_dir, args.max_steps)
        return

    if args.array_id is not None:
        if not 0 <= args.array_id < len(tasks):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        c, f, d, s = tasks[args.array_id]
        print(f"[array {args.array_id}/{len(tasks)}] collapse={c} fraction={f} d_model={d} seed={s}")
        run_one(c, f, d, s, args.output_dir, args.max_steps)
        return

    for i, (c, f, d, s) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] collapse={c} fraction={f} d_model={d} seed={s}\n{'='*60}")
        run_one(c, f, d, s, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
