# Phase 4: Mechanistic Interpretability Reproduction

This document describes how to run the mechanistic interpretability tools developed to study the grokking vs. collapse phenomenon in the Transformer models.

## Environment

Ensure the environment is set up and active:
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt # or manually install torch numpy matplotlib scipy pytest
```

## Running Tests

All the analysis and experiment modules are covered by unit tests. You can run them via pytest:
```bash
pytest tests/
```

## Usage Guidelines

1. **Circuit Analysis (`analysis/circuits.py`)**:
   - Extract attention patterns using `extract_attention_patterns(model, x)`.
   - Identify heads that specialize after grokking via `identify_grokking_circuits`.

2. **Weight Space Analysis (`analysis/weight_space.py`)**:
   - Compute effective rank, weight norms, and top Hessian eigenvalues on checkpoints to see how grokking restructures the model weight geometry compared to the collapsed model geometry.

3. **Gradient Flow Analysis (`analysis/gradient_flow.py`)**:
   - Use `approximate_gradients(model_prev, model_curr)` to inspect the direction of learning.
   - Use `identify_gradient_starvation` to detect parameters that cease updating in collapsed runs.

4. **Interventions (`experiments/interventions.py`)**:
   - Run causal tests using `run_head_ablation` or freezing specific layers with `run_weight_freezing` during training to confirm if specific structures are critical for grokking.
