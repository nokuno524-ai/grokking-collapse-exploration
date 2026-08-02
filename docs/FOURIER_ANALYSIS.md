# Fourier and Weight Distribution Analysis

This repository includes advanced analysis modules to investigate the mechanisms of model grokking and collapse, specifically utilizing Fourier analysis and weight distribution tracking.

## Fourier Analysis (`src/analysis/fourier.py`)

During grokking on algorithmic tasks like modular arithmetic, the model often forms clear, sparse circuits in the Fourier domain. We track this evolution using 2D Fourier transforms of weight matrices.

### Key Metrics
1. **2D Fourier Transform**: Computes the 2D magnitude spectrum of an arbitrary 2D weight matrix using `torch.fft.fft2`. We use orthogonal normalization to maintain energy scaling.
2. **Fourier Concentration**: Evaluates how much of the weight's structure is explained by the top frequencies. We compute the fraction of the total spectral energy contained in the top $k$ frequencies, strictly excluding the DC component (at index `[0, 0]`). This allows us to track circuit formation independently of the mean weight values.
   - Formula: $C_k = \frac{\sum_{i=1}^k E_{(i)}}{\sum_{j \neq DC} E_j}$

### Visualizations
- `plot_fourier_heatmap`: Generates heatmaps of the 2D Fourier magnitude spectrum using a logarithmic scale (`np.log1p`) for clearer visualization of frequency spikes.

## Weight Distribution Analysis (`src/analysis/weights.py`)

Model collapse often correlates with stark changes in weight geometry and sparsity. We track these changes through comprehensive summary statistics and effective rank measures.

### Key Metrics
1. **Distribution Statistics**:
   - **Kurtosis**: Measures the "tailedness" of the weight distribution. We use Fisher's definition where a normal distribution has a kurtosis of 0. High kurtosis indicates heavier tails and more outliers.
   - **Skewness**: Measures the asymmetry of the weight distribution.
   - **Sparsity**: Measures the fraction of weight elements whose absolute value falls below a defined `sparsity_threshold`.
2. **Effective Rank**: Computes the rank of a weight matrix smoothly via the Shannon entropy of its normalized singular values.
   - Formula: $H = - \sum \tilde{\sigma_i} \log \tilde{\sigma_i}$, where $\tilde{\sigma_i} = \frac{\sigma_i}{\sum \sigma_j}$.
   - Effective Rank = $\exp(H)$

### Visualizations
- `plot_weight_histogram`: Plots a histogram of weight values overlaid with kurtosis and skewness metrics.

## Correlation Study (`src/analysis/correlations.py`)

A core objective is to identify leading indicators for grokking. The module provides functionality to correlate early training phase signals with final grokking success.
- **Early Fourier Correlation**: Correlates the average Fourier concentration up to a specific early step (e.g., step 1000) with both the occurrence of grokking (success/failure) and the actual step at which grokking occurs.

## Usage
These functions are designed to be imported within analysis scripts and applied to extracted model weights at different checkpoint intervals.

Example:
```python
from src.analysis.fourier import compute_2d_fourier_transform, compute_fourier_concentration
from src.analysis.weights import compute_weight_statistics, compute_effective_rank

spectrum = compute_2d_fourier_transform(model_weights)
concentration = compute_fourier_concentration(spectrum, top_k=5)
stats = compute_weight_statistics(model_weights)
rank = compute_effective_rank(model_weights)
```
