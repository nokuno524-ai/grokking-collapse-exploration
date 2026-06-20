# Mechanistic Analysis of Grokking under Collapse

This document outlines the mechanistic analysis toolkit used to investigate how dataset collapse prevents grokking at the circuit level. By utilizing techniques from mechanistic interpretability, we track the formation, density, and specialization of sub-networks over training.

## Toolkit Overview

The analysis toolkit is broken down into four independent modules in `analysis/`, all tracking metrics across training trajectories and comparing between "pure" and "collapsed" models.

### 1. Induction Head Formation (`induction_heads.py`)
Since the dataset is `(a, b)` -> `(a+b) mod p`, there are no standard repeating sequences. Instead, we adapt the prefix-matching concept to look for heads that form a deterministic "identity" or "permutation" mapping by overwhelmingly attending to a specific token. We track the **maximum attention sharpness** (how heavily a head focuses on a single position) as a proxy for the formation of structured computation paths, comparing when these paths solidify relative to the grokking point.

### 2. Circuit Complexity (`circuit_complexity.py`)
We measure the overall density and complexity of the model's internal routing. A simple activation attribution proxy (the scaled magnitude of the key/query/value matrices) is used. Specifically, we calculate a **participation ratio** of the attention head outputs.
- A sparse participation ratio suggests highly specialized, distinct circuits (expected in pure grokking).
- A dense participation ratio suggests disorganized, polysemantic entanglement (expected under severe collapse).

### 3. Neuron-Level Specialization (`neuron_analysis.py`)
We evaluate the activation patterns of the feedforward network (FFN). Grokking has been associated with the formation of highly specific, monosemantic neurons. We approximate **polysemanticity** using the kurtosis (inverse sparsity) of neuron activations over the entire dataset.
- Highly localized, spiky activations indicate specialized functions.
- Uniform, distributed activations indicate polysemantic entanglement.

### 4. Information Flow (`info_flow.py`)
We trace how and where information about the target label is represented throughout the layers of the network (post-embedding, post-attention, post-FFN). Since exact mutual information is intractable, we use a **linear probing accuracy proxy**. We fit simple linear logistic regression probes at each stage to map out where information bottlenecks occur and when the labels become linearly decodable.

## Running the Analyses

All tools can be run independently as standalone scripts, expecting results to be stored in the default `results/exp_c_grid/` directory.

```bash
python analysis/induction_heads.py
python analysis/circuit_complexity.py
python analysis/neuron_analysis.py
python analysis/info_flow.py
```

The output for each tool is a timeline plot saved directly to the `analysis/` folder, allowing visual inspection of how mechanisms form under different levels of data contamination.
