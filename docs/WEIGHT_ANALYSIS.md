# Weight-Space Analysis Tools

This repository contains tools to perform deep weight-space analysis to understand how and why weights degenerate during model collapse, and how this geometry impacts generalization (grokking).

## 1. Weight Norm Evolution (`analysis/weight_evolution.py`)
### Theoretical Background
Prior observations indicate that collapsed models exhibit a significant reduction in weight norm (30-42%). Weight decay acts differently on collapsed vs. grokked models. Tracking the L2 norm of weights across training helps identify which layers collapse first and whether norm shrinkage is a reliable leading indicator of generalization failure.
### Usage
- `compute_layer_norms(state)`: Extracts L2 norm of weight matrices.
- `track_norm_trajectories(checkpoints)`: Returns norm history.
- `compare_decay_paths(...)`: Plots grokked vs collapsed weight norm trajectories.
- `check_norm_reduction_predictor(...)`: Statistical significance test for weight norm reduction.

## 2. Hessian Analysis (`analysis/hessian.py`)
### Theoretical Background
The geometry of the loss landscape defines optimization stability. Flatter minima (lower sharpness / max Hessian eigenvalue) correlate strongly with better generalization. Under collapse conditions, we track the top eigenvalues to see if the model converges to sharper minima.
- We use **Power Iteration with Deflation** as a memory-efficient alternative to full Lanczos for estimating the top-K eigenvalues without materializing the full Hessian matrix.
### Usage
- `power_iteration(...)`: Finds the maximum eigenvalue (sharpness).
- `compute_top_k_eigenvalues(...)`: Finds multiple top eigenvalues to estimate local dimension.
- `estimate_hessian_rank(...)`: Uses a threshold on eigenvalues.
- `track_sharpness(...)` & `compare_landscape_geometry(...)`: Visualizing sharpness over time.

## 3. Loss Landscape Visualization (`analysis/loss_landscape.py`)
### Theoretical Background
Visualizing high-dimensional loss landscapes is challenging due to scaling issues across different layers. We implement **filter normalization** (Li et al., 2018) to ensure random direction vectors used for contour plots scale correctly relative to the filter's existing weight norm, producing meaningful contours.
### Usage
- `interpolate_1d(...)`: Linear interpolation between two states (e.g. init and final, or grokked and collapsed).
- `plot_2d_landscape(...)`: Generates a filter-normalized 2D contour plot around a given state.
- `overlay_trajectory(...)`: Projects a sequence of optimization steps onto the 2D plane defined by the contour plot's basis vectors.

## 4. Representational & Weight Similarity (`analysis/weight_similarity.py`)
### Theoretical Background
Centered Kernel Alignment (CKA) is a robust similarity measure that is invariant to orthogonal transformations. We apply CKA to the flattened weight matrices to track how the functional structure of the weights diverges between grokked models and models undergoing collapse.
### Usage
- `compute_cka(A, B)`: Computes linear CKA between two matrices.
- `compute_weight_cka(state_a, state_b)`: Computes CKA layer-by-layer.
- `track_similarity_trajectory(...)`: Plots CKA over time against a reference state.

## Integration & Analysis Pipeline
These tools are designed to work together with synthetic checkpoints saved by the main training loop (`src/train.py`). Typically, analysis scripts will load a list of checkpoint dictionaries containing `model_state`, feed them to trajectory-tracking functions, and generate comparative plots for a unified report on model collapse mechanisms.