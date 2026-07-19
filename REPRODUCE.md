# Reproducing Grokking & Model Collapse Experiments

This guide provides the exact commands to recreate the findings and analyses presented in this repository.
All experiments were run with a fixed random seed of `42` unless stated otherwise.

## 1. Environment Setup

We use `uv` for fast python environment management.

```bash
# Create a virtual environment
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install all necessary dependencies
uv pip install pyyaml torch numpy matplotlib scipy pandas seaborn pytest tabulate hydra-core plotly black isort flake8 mypy statsmodels pydantic scikit-learn tensorboard
```

## 2. Generating the Base Results (Training)

If the `results/` folder is empty, you must first train the models.
The script `run_experiment.sh` coordinates training across multiple collapse conditions.

```bash
# Run the training loop for pure and collapse conditions
./run_experiment.sh
```
*Note: This will execute training for conditions: `pure`, `low`, `medium`, `severe`, and `high` collapse using `src/train.py`.*

## 3. Analysis and Visualizations

After training, you can run the suite of mechanistic and statistical analysis tools.

### Attention Pattern Evolution
To generate interactive HTML animations of how attention maps evolve:
```bash
python analysis/attention_evolution.py
```
*Output: `results/attention_evolution_*.html`*

### Circuit Formation (Activation Patching / Ablation)
To measure the importance of specific attention heads over time:
```bash
python analysis/circuit_formation.py
```
*Output: `results/circuit_formation_*.json` and `.png`*

### Loss Landscape Visualization
To visualize the 1D loss landscape around trained checkpoints via filter normalization:
```bash
python analysis/loss_landscape.py
```
*Output: `results/loss_landscape_1d_*.png`*

### Phase Transition Detection
To dynamically detect the exact step where grokking occurs using the accuracy derivative:
```bash
python analysis/phase_transition.py
```
*Output: `results/phase_transitions.json`*

### Summary Metrics Parsing
To parse the metrics across all conditions and generate a comparative multi-panel figure:
```bash
python analysis/parse_results.py
```
*Output: `results/parsed_summary.csv` and `results/summary_multi_panel.png`*

## 4. Running the Tests
To ensure the analytical functions operate correctly without running the entire training pipeline:

```bash
pytest tests/
```
