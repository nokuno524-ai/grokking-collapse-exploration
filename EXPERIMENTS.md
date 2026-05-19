# Experiment Management

This repository uses a structured configuration system to manage single and sweep (grid) experiments. All configuration is defined in YAML files.

## Running a Single Experiment

1. **Create a YAML config file (`config.yaml`)**:
```yaml
experiment_name: "my_baseline"
model:
  prime: 59
  d_model: 128
dataset:
  train_fraction: 0.3
  collapse_level: 0.0
training:
  max_steps: 50000
  weight_decay: 1.0
  lr: 0.001
  seed: 42
```

2. **Run it locally**:
```bash
python run_experiment.py --config config.yaml
```
This will create an output directory under `results/<experiment_name>_<timestamp>_<uuid>/` storing the model checkpoints, `config.json`, and `results.json`. The results JSON includes hashes of the dataset and the exact Git commit used.

## Running a Parameter Sweep

1. **Create a sweep config file (`sweep.yaml`)**:
```yaml
experiment_name: "wd_sweep"
base_config:
  model:
    prime: 59
  dataset:
    train_fraction: 0.3
  training:
    max_steps: 50000
    seed: 42
sweep_params:
  training.weight_decay: [0.1, 1.0, 3.0]
  dataset.collapse_level: [0.0, 0.15, 0.3]
```

2. **Run the sweep locally**:
```bash
python run_experiment.py --config sweep.yaml --sweep
```

## Submitting to Slurm

You can easily generate and submit SLURM sbatch scripts for either single configs or sweeps using `submit_sbatch.py`.

```bash
# Generate and submit a job
python submit_sbatch.py --config sweep.yaml --sweep --partition gpu --gpu a100:1 --cpus 4 --memory 16G

# Or just test generating the script without submitting
python submit_sbatch.py --config config.yaml --dry-run
```
Logs for Slurm runs will be placed in `slurm_logs/`.

## Analyzing Results (Dashboard)

A standalone dashboard generator will parse all `results.json` files recursively across your experiments directory and produce an HTML visualization:

```bash
# Set PYTHONPATH to root so it can import src
PYTHONPATH=. python tools/dashboard.py --results-dir results --output dashboard.html
```

The `dashboard.html` file will embed static Matplotlib visualizations comparing accuracy and phase transition timings across the configurations, and give you an overview of Slurm queue statuses.
