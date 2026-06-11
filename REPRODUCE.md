# Reproducibility Guide

This guide details how to reproduce the scaling experiments and phase detection analysis for the interplay between model collapse and grokking.

## Environment Setup

The repository uses `uv` for fast environment setup, but standard `pip` is also supported. Exact package versions are pinned in `requirements.txt`.

### Using the Automated Script

We provide a single `run_all.sh` script that sets up the environment, runs all tests, and executes the full suite of scaling experiments.

```bash
chmod +x run_all.sh
./run_all.sh
```

### Manual Setup

If you prefer to run steps manually:

1. Create and activate a virtual environment:
```bash
uv venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Run the test suite to verify the novel collapse metrics and phase detection logic:
```bash
PYTHONPATH=. pytest tests/
```

4. Execute the scaling experiments:
```bash
python -c "
import sys; sys.path.append('.')
from experiments.scaling import run_scaling_experiments, plot_scaling_laws, ScalingExperimentConfig
config = ScalingExperimentConfig(max_steps=10000, eval_every=500)
results = run_scaling_experiments(config)
plot_scaling_laws(results, 'results/scaling/plots')
"
```

## Expected Outputs

After running the experiments, you should find the following in the `results/scaling/` directory:

1. `runs/`: Subdirectories containing `results.json` and checkpoints for every `(prime, d_model, collapse_level)` configuration.
2. `scaling_results.json`: A consolidated JSON file mapping `prime -> d_model -> collapse_level -> {grokked, grok_step, final_test_acc}`.
3. `plots/`: Scaling law plots (`scaling_law_p29.png`, etc.) showing the grokking step vs model size for each collapse level.

## Verification

You can verify the correctness of the novel metrics implemented in `src/collapse_metrics.py` and the automated detection logic in `src/phase_detection.py` by ensuring that all tests pass (`pytest tests/`). The tests include validations against constructed accuracy curves and known synthetic matrices.
