# Statistical Analysis of Grokking and Model Collapse

This document outlines the methodology, usage, and interpretation of the statistical analysis tools added to evaluate the differences between grokking in pure data conditions versus model collapse conditions.

## Methodology

### 1. Statistical Tests
To ensure the robustness of the experimental findings, several statistical tests have been implemented in `analysis/statistical_tests.py`:
- **Welch's t-test**: Used to compare the final test accuracy between independent conditions (e.g., pure vs. severe collapse). Welch's t-test does not assume equal variances.
- **Kolmogorov-Smirnov (KS) Test**: Used to compare the continuous distributions of final weight norms between conditions, testing whether they are drawn from the same underlying distribution.
- **Bootstrap Confidence Intervals (BCa)**: Used to compute 95% confidence intervals for the onset step of grokking. The implementation robustly handles degenerate cases where variance is zero.
- **Cohen's d Effect Size**: Quantifies the magnitude of difference between two groups (e.g., the final test accuracy between pure and collapsed models).

### 2. Experimental Sweeps and Reproducibility
The `reproduce.py` script executes a comprehensive sweep across all collapse severity conditions (`pure`, `low_collapse`, `medium_collapse`, `high_collapse`, `severe_collapse`).
- **Seeding**: Deterministic behavior is enforced by strictly setting seeds for `torch`, `numpy`, and `random` prior to each model initialization and data generation step.

## Reproducing the Results

To run the reproducibility sweep (e.g., across 5 random seeds):

```bash
# Activate your virtual environment if not already active
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# Run the full sweep (Note: --max-steps is set to 50000 by default for full training, but can be overridden)
python reproduce.py --seeds 42 43 44 45 46 --max-steps 50000 --output-dir results/reproduce
```

## Analyzing the Results

After the `reproduce.py` script has generated the raw data, use `run_analysis.py` to process the `results.json` files and generate the statistical report, LaTeX summary tables, and publication-quality figures.

```bash
python run_analysis.py --results-dir results/reproduce --output-dir analysis_output
```

This will output the following files in the `analysis_output/` directory:
- `summary_table.tex`: A LaTeX table summarizing test accuracy, weight norm, Fourier concentration, and grokking steps with confidence intervals.
- `statistical_tests.md`: A markdown file containing the output of the Welch's t-test, Cohen's d, and KS tests.
- `test_accuracy_boxplot.png`: A box plot visualizing the distribution of final test accuracy across all severity levels.
- `weight_vs_fourier.png`: A scatter plot detailing the relationship between weight norm and Fourier concentration, colored by condition.
- `correlation_matrix.png`: A heatmap of Pearson correlations between collapse level, test accuracy, weight norm, and Fourier concentration.
