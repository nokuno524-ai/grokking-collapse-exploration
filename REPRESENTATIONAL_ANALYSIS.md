# Representational Analysis Toolkit

This document describes the methodology, interpretation guide, and connections to grokking theory for the representational geometry and circuit-level analysis tools implemented in this repository.

## 1. Representational Geometry Analysis (`src/representational_analysis/geometry.py`)

We examine how the model's internal representation geometry evolves during training, with a specific focus on the phase transition (grokking).

### Methodologies
* **Representational Similarity Analysis (RSA):** Computes Representational Similarity Matrices (RSMs) using cosine, euclidean, or correlation metrics. Tracking RSM similarity over checkpoints reveals when the model's internal organization of data shifts (e.g., from rote memorization to structural comprehension).
* **Intrinsic Dimensionality (ID):**
  * **Participation Ratio (PR):** Measures the continuous effective rank of the activation covariance matrix. A drop in PR often accompanies the generalization phase.
  * **Maximum Likelihood Estimation (MLE):** Uses nearest-neighbor distances to estimate the local manifold dimension.
* **Neural Anisotropy:** Evaluates the alignment of representations with specific, task-relevant directions. Increased alignment indicates the model is separating features along functional axes.
* **Centered Kernel Alignment (CKA):** Compares representational structures across different layers or different training steps, invariant to orthogonal transformations and isotropic scaling.

## 2. Circuit-Level Analysis (`src/representational_analysis/circuit.py`)

Following insights that circuits reorganize during grokking, these tools analyze the causal and functional role of specific model components.

### Methodologies
* **Attention Head Importance:** Uses causal patching (zeroing out a head's contribution via its `out_proj.weight`) to measure the drop in accuracy when a head is ablated.
* **Tracking Importance Over Time:** By evaluating head importance across training checkpoints, we can identify which heads become critical during the phase transition.
* **MLP Neuron Analysis:** Uses forward hooks to capture feed-forward network (FFN) activations. Calculates mean activation, variance, and kurtosis to identify highly selective (task-specific) neurons.
* **Circuit Emergence Detection:** Tracks changes in inter-head communication or attention scores to identify the exact step when functional circuits stabilize.

## 3. Representational Intervention (`src/representational_analysis/intervention.py`)

These tools test whether structure-specific priors causally control the grokking delay.

### Methodologies
* **Activation Steering:** Uses PyTorch forward hooks to inject specific representational structures (steering vectors) at specific layers during the forward pass.
* **Weight Structure Intervention:** A regularizer added to the training loss that encourages weights to maintain geometric properties associated with grokking (e.g., target rank or high Fourier concentration).
* **Geometry Pre-initialization:** Initializes embedding weights with grokking-favorable geometry (like Fourier basis components) to test if this accelerates grokking onset.

## 4. Phase Transition Detection (`src/representational_analysis/phase_transition.py`)

To precisely identify grokking onset, we use statistical tools on training metrics (like test accuracy or loss).

### Methodologies
* **Change-Point Detection:** Uses rolling window derivatives to find significant jumps in metrics, typically characterizing the start of the grokking phase.
* **Derivative Metrics:** Computes first (velocity) and second (acceleration) derivatives of metrics with respect to training steps.
* **Piecewise Linear Regression:** A statistical test for phase transitions using regression with a single breakpoint, allowing us to find the exact step where the metric trajectory fundamentally changes.

## Connection to Grokking Theory

Grokking is typically observed as a sharp transition from memorization (high train accuracy, low test accuracy) to generalization (high test accuracy). The tools provided here map to theoretical explanations of this phenomenon:
1. **Representation Compression:** The transition is often accompanied by a decrease in intrinsic dimensionality (measured via PR and MLE) and an increase in structured representations (measured via RSA and CKA).
2. **Circuit Formation:** Generalization relies on the formation of robust, task-relevant circuits. Our circuit analysis tools track when and how these circuits emerge.
3. **Causal Control:** By intervening in representations (steering, regularization), we test the hypothesis that specific geometries (like Fourier concentration) are not just byproducts of grokking, but causal drivers that can accelerate or delay it.
