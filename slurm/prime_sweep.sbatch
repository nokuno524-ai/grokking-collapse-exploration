#!/bin/bash
#SBATCH --job-name=grok-primesw
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/primesw-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/primesw-%A_%a.err
#SBATCH --array=0-89%30

# Prime-brittleness sweep: p in {59, 97, 113} x wd=1.0 x noise in {0,.05,.1,.15,.2,.3} x 5 seeds = 90.
# p=113 runs take ~2x as long as p=59 (dataset is 4x bigger) but still fits in 90 min.

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "[prime-sweep] task=${IDX} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

python3 -m src.run_prime_sweep \
    --array-id "${IDX}" \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/prime_sweep \
    --max-steps 50000

echo "[prime-sweep] task=${IDX} done"
