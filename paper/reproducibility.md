# Reproducibility Details

This document outlines the requirements and commands necessary to reproduce the experimental results presented in the paper.

## Hardware Requirements
- **GPU:** A single NVIDIA GPU (e.g., A100, V100, RTX 3090, or RTX 4090) with at least 8GB of VRAM is sufficient for the toy experiments. The real-LM contamination experiments (if executed) require at least 24GB VRAM.
- **CPU:** A standard modern multi-core CPU.
- **RAM:** 16GB system RAM.
- **Disk Space:** Approximately 20GB for checkpoints and logs, mostly in `results/`.

## Environment Setup
The project uses `uv` for dependency management and environment creation. Python 3.10+ is recommended.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch numpy matplotlib scipy pandas seaborn pytest tabulate
```

## Running the Experiments

### 1. The Core Grid Sweep ($wd \times \eta$)
This is the primary experiment to generate the phase diagram and measure the grokking cliff. We use a SLURM script to dispatch the grid across 90 jobs (3 wd $\times$ 6 noise $\times$ 5 seeds).

```bash
# Submit to SLURM array
sbatch slurm/exp_c_grid.sbatch
```

If running locally (not recommended sequentially due to time), you would invoke:
```bash
python src/train.py --condition pure --weight-decay 1.0 --noise-fraction 0.05 --seed 42 --max-steps 50000
```
*(Repeat for each combination of wd, noise, and seed).*

**Expected Runtime:** Each single toy training run takes approximately 10-15 minutes on an A100 GPU. The full grid sequentially would take ~20 hours.

### 2. Baselines (Noise, Scarcity, Multi-seed)
To reproduce the specific baselines:
```bash
sbatch slurm/baselines.sh
```

### 3. Surgical Transplants (Mitigation)
Once the grid sweep is completed, the zero-shot transplants can be run to evaluate missing circuit components. Example command for a specific seed pair (pure vs contaminated):

```bash
python src/transplant_rescue.py \
    --pure-run results/exp_c_grid/wd1.0/noise0.0/seed_42 \
    --contam-run results/exp_c_grid/wd1.0/noise0.15/seed_42 \
    --output-dir analysis/transplant
```

**Expected Runtime:** $\sim$5 minutes per transplant pair.

### 4. Analysis and Figure Generation
The analysis scripts parse the generated `results/` and compile the data into `analysis/`. Our unified paper figure generation script aggregates these into the final PDF/PNG plots.

```bash
python analysis/exp_c_grid_analysis.py
python scripts/generate_paper_figures.py
```

## Random Seeds
All experiments are repeated over 5 random seeds to ensure statistical significance. The default seeds used in the grid sweep are `[42, 43, 44, 45, 46]`. For PyTorch reproducibility, seeds are set explicitly at the beginning of `src/train.py` via:
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `numpy.random.seed(seed)`
- `random.seed(seed)`
