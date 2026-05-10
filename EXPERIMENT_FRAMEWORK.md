# Experiment Framework Guide

The modular experiment framework aims to make managing, configuring, running, and analyzing experiments relating to LLM grokking and collapse simple and reproducible.

## Configuration System

The configuration system relies on Python dataclasses from `src/experiments/config.py`.

- `ExperimentConfig`: Defines hyperparameters (learning rate, epochs, weight decay, model type, seed).
- `CollapseConfig`: Specifically isolates the injection mechanism for simulating collapse (e.g., `weight_noise`, `synthetic_data_ratio`).

### Saving and Loading Configs

You can serialize configurations to JSON or YAML:

```python
from src.experiments.config import ExperimentConfig, save_config, load_config

# Create a config
config = ExperimentConfig(epochs=1000, collapse_level=0.5)

# Save to file
save_config(config, "my_config.yaml")

# Load from file
loaded_config = load_config("my_config.yaml")
```

## Running Experiments

`ExperimentRunner` from `src/experiments/runner.py` manages the actual training process. It will automatically initialize the model and standard datasets deterministically, apply any necessary collapse interventions throughout training, log intermediate metrics, and finally save everything.

```python
from src.experiments.config import load_config
from src.experiments.runner import ExperimentRunner

config = load_config("my_config.yaml")
runner = ExperimentRunner(config)

# Start training
results = runner.run()
```

## Weight Analysis

Understanding the structural changes in model weights across grokking and collapse relies on modules in `src/analysis/weight_analysis.py`.

```python
from src.analysis.weight_analysis import (
    compute_weight_norms,
    compute_weight_rank,
    track_weight_evolution,
    detect_collapse_from_weights,
    plot_weight_analysis
)

# ... inside your evaluation logic ...
norms = compute_weight_norms(model)
ranks = compute_weight_rank(model, threshold=0.99)

# To analyze over time, maintain snapshots of the model
# history = track_weight_evolution(model_snapshots)
# collapse_signatures = detect_collapse_from_weights(history)

# Save text-based analysis and minimal plots (if matplotlib installed)
# plot_weight_analysis(history, "results/weight_analysis_output")
```
