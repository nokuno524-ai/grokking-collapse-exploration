import os
from pathlib import Path
from .config import ExperimentConfig

class SlurmGenerator:
    """Generates Slurm sbatch scripts for experiment grids."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def generate_script(self, output_path: str, runner_script: str = "src/management/runner.py") -> str:
        """
        Generates the content of the Slurm script and saves it.
        Returns the content as a string.
        """
        num_tasks = self.config.get_num_tasks()
        array_str = f"0-{num_tasks - 1}" if num_tasks > 1 else "0"

        # Determine paths
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        out_log = log_dir / f"{self.config.name}-%A_%a.out"
        err_log = log_dir / f"{self.config.name}-%A_%a.err"

        config_path = Path(self.config.output_dir) / f"{self.config.name}_config.yaml"

        # Base template
        script = f"""#!/bin/bash
#SBATCH --job-name={self.config.name}
#SBATCH --account={self.config.compute.account}
#SBATCH --partition={self.config.compute.partition}
#SBATCH --gres=gpu:{self.config.compute.gpus}
#SBATCH --cpus-per-task={self.config.compute.cpus_per_task}
#SBATCH --mem={self.config.compute.mem}
#SBATCH --time={self.config.compute.time}
#SBATCH --array={array_str}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

set -euo pipefail
export PYTHONUNBUFFERED=1

# Change to repo root
cd $(dirname $(dirname $(dirname $(realpath $0)))) || cd .
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

IDX=${{SLURM_ARRAY_TASK_ID:-0}}
echo "[slurm] {self.config.name} task ${{IDX}}/{num_tasks} on $(hostname)"

# Run the runner script
python {runner_script} \\
    --config-path {config_path} \\
    --array-id "${{IDX}}"
"""

        # Make sure directory exists before saving script
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(script)

        return script
