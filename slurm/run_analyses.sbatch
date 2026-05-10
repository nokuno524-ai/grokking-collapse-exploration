#!/bin/bash
#SBATCH --job-name=grok-analyses
#SBATCH --account=zhangmlgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/analyses-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/analyses-%j.err

# Runs all post-hoc analyses on the math grokking project:
#   1. analysis/exp_c_grid_analysis.py    -> heatmaps, trajectory diag, fourier cliff
#   2. analysis/comprehensive_analysis.py -> toy + real combined report
#   3. src/causal_circuit_rescue.py       -> per-matrix rank trajectories
# All outputs land under analysis/.

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

echo "[analyses] step 1/3: exp_c_grid_analysis"
python3 analysis/exp_c_grid_analysis.py

echo "[analyses] step 2/3: comprehensive_analysis"
python3 analysis/comprehensive_analysis.py

echo "[analyses] step 3/3: causal_circuit_rescue"
python3 -m src.causal_circuit_rescue \
    --runs results/exp_c_grid/wd0.3/noise0/seed_42 \
           results/exp_c_grid/wd0.3/noise0.05/seed_42 \
           results/exp_c_grid/wd0.3/noise0.15/seed_42 \
           results/exp_c_grid/wd1/noise0/seed_42 \
           results/exp_c_grid/wd1/noise0.15/seed_42 \
           results/exp_c_grid/wd3/noise0/seed_42 \
    --output-dir analysis/causal_circuit

echo "[analyses] all done"
