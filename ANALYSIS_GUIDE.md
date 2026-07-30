# Analysis Guide: Reproducing the Results

This guide provides step-by-step instructions for running the mechanistic interpretability and statistical analysis suite.

## 1. Environment Setup

Ensure you are using the `uv` environment as prescribed:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt # Ensure torch, numpy, scipy, matplotlib, seaborn, pandas, pytest are installed
```
*(If `requirements.txt` does not exist, simply `uv pip install torch numpy scipy matplotlib seaborn pandas pytest`)*

## 2. Generating the Data (Training)

If you have not already trained the models across collapse conditions:
```bash
# Example training runs for different conditions
python src/train.py --condition pure --max-steps 10000
python src/train.py --condition low_collapse --max-steps 10000
# (Outputs will be saved to the results/ directory)
```

## 3. Running the Analysis Modules

The analysis suite is located in `src/analysis/`. You can import these functions in your custom scripts or notebooks to analyze checkpoints.

### Statistical Analysis
To compute confidence intervals or test significance between grokking steps:
```python
from src.analysis.statistics import permutation_test_grokking, bootstrap_ci

# Example: Compare grokking steps between pure and low_collapse seeds
diff, p_val = permutation_test_grokking(pure_steps, low_collapse_steps)
```

### Weight & Geometry Analysis
To track weight norms or landscape sharpness:
```python
from src.analysis.weight_analysis import get_weight_norms, compute_hessian_max_eigenvalue

norms = get_weight_norms(model)
eig = compute_hessian_max_eigenvalue(model, dataloader, criterion)
```

### Mechanistic Interpretability
To perform activation patching or Fourier tracking:
```python
from src.analysis.interpretability import activation_patching, track_fourier_coefficients

# Patch linear1 from pure to collapsed model
patched_out = activation_patching(pure_model, collapsed_model, 'linear1', inputs)

# Get Fourier spectrum of embeddings
fourier = track_fourier_coefficients(model)
```

## 4. Generating the Dashboard

To produce the unified multi-panel publication figure:
```python
from src.analysis.dashboard import generate_comparison_dashboard

# Prepare your aggregated data dictionaries...
generate_comparison_dashboard(
    loss_data,
    weight_norm_data,
    attention_entropy_data,
    grokking_prob_matrix,
    collapse_levels=['pure', 'low', 'medium', 'severe', 'high'],
    model_sizes=['small', 'medium', 'large'],
    fourier_spectra=fourier_data,
    save_path="results/dashboard/comparison_dashboard.pdf"
)
```

## 5. Testing

To verify the integrity of the analysis suite, run the test suite:
```bash
pytest tests/test_analysis.py
```
This runs fast tests using synthetic models and data to ensure the math and tensor operations are correct.
