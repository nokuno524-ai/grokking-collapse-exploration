# Reproducibility Guide

This document details the precise commands and configurations required to reproduce the key findings using the automated experiment runner.

## Environment Setup

1. Create and activate a Python virtual environment:
```bash
uv venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r reproduce/requirements.txt
# Alternatively, to capture all dependencies:
uv pip install torch numpy matplotlib seaborn pytest pyyaml hydra-core pandas pydantic
```

## Running the Automated Experiments

The automated grid search experiments span multiple parameter combinations (model sizes, dataset composition ratios, collapse severities, random seeds, and training steps). The configurations are stored in YAML format.

1. **Create an `experiments.yaml` file:**
   ```yaml
   model_size:
     d_model: 128
     n_heads: 4
     n_layers: 1
   dataset:
     prime: 59
     train_fraction: 0.3
   composition_ratios: [0.0, 0.05, 0.15, 0.30, 0.50]
   collapse_severities: [0.3, 0.5, 0.7, 0.9]
   training_steps: 50000
   seeds: [42, 43, 44, 45, 46]
   output_dir: "results_automated"
   ```

2. **Execute the Experiment Runner:**
   Run the experiments sequentially or in parallel depending on your hardware.

   To run in parallel (recommended for multi-core CPUs):
   ```bash
   export PYTHONPATH=$(pwd)
   python src/grokking/run_experiments.py --config experiments.yaml --parallel --workers 8
   ```

   **Hardware Requirements & Runtime:**
   - **Hardware**: For parallel execution with 8 workers, an 8+ core CPU and 32GB+ RAM are recommended. If using CUDA, multiple GPUs or a single large GPU (e.g. A100 or RTX 4090) is necessary.
   - **Runtime**: Depending on the exact number of steps and the number of workers, a full 5-seed grid of 20 conditions (100 runs total) can take several hours. The runner supports graceful resumption: if interrupted, re-running the command skips already completed conditions.

## Result Aggregation

Once the experiments complete, aggregate the raw `.json` outputs to generate CSV summaries and publication-ready plots.

```bash
export PYTHONPATH=$(pwd)
python src/grokking/aggregate_results.py --output-dir results_automated --summary-file summary.csv
```

**Expected Outputs:**
- `results_automated/summary.csv`: A flattened CSV table computing the mean and standard deviation of `test_accuracy`, `train_accuracy`, `weight_norm`, `embedding_rank`, and `fourier_concentration` grouped by collapse conditions.
- `results_automated/test_accuracy.png`: Seaborn plot of Test Accuracy over time per collapse level.
- `results_automated/weight_norm.png`: Seaborn plot of L2 Weight Norms over time per collapse level.
