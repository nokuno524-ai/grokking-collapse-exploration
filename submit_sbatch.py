#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

# Template for sbatch script
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
{gpu_line}

# Unbuffered output
export PYTHONUNBUFFERED=1

# Activate virtual environment
source .venv/bin/activate

echo "Starting job: {job_name}"
echo "Running config: {config_path}"

# Run the experiment
python run_experiment.py --config {config_path} {sweep_flag}

echo "Job complete."
"""

def generate_sbatch(args, config_path, sweep_flag=""):
    """Generate the SBATCH script content."""
    job_name = Path(config_path).stem

    # We create a specific slurm output directory for this job run
    slurm_dir = Path("slurm_logs")
    slurm_dir.mkdir(parents=True, exist_ok=True)

    gpu_line = f"#SBATCH --gres=gpu:{args.gpu}" if args.gpu else ""

    script_content = SBATCH_TEMPLATE.format(
        job_name=job_name,
        output_dir=str(slurm_dir),
        partition=args.partition,
        cpus=args.cpus,
        memory=args.memory,
        time=args.time,
        gpu_line=gpu_line,
        config_path=config_path,
        sweep_flag=sweep_flag
    )

    return script_content

def main():
    parser = argparse.ArgumentParser(description="Submit Grokking-Collapse experiments to Slurm.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--sweep", action="store_true", help="Treat config as a SweepConfig")
    parser.add_argument("--partition", type=str, default="gpu", help="Slurm partition")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task")
    parser.add_argument("--memory", type=str, default="16G", help="Memory request")
    parser.add_argument("--time", type=str, default="24:00:00", help="Time limit")
    parser.add_argument("--gpu", type=str, default="a100:1", help="GPU specification (e.g. a100:1). Leave empty for CPU only")
    parser.add_argument("--dry-run", action="store_true", help="Generate script but do not submit")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    sweep_flag = "--sweep" if args.sweep else ""
    sbatch_script = generate_sbatch(args, args.config, sweep_flag)

    script_path = Path(f"{Path(args.config).stem}.sbatch")
    with open(script_path, "w") as f:
        f.write(sbatch_script)

    print(f"Generated SLURM script: {script_path}")

    if args.dry_run:
        print("Dry run requested. Not submitting.")
        print("Script contents:")
        print("-" * 40)
        print(sbatch_script)
        print("-" * 40)
    else:
        print("Submitting to SLURM...")
        try:
            result = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit to SLURM: {e.stderr}")
            # If sbatch is not available, just warn
            if "No such file or directory" in str(e) or "command not found" in e.stderr.lower() or "not found" in str(e):
                print("Warning: 'sbatch' command not found. Are you on a SLURM cluster?")

if __name__ == "__main__":
    main()
