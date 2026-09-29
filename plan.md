1. **Code Review**: I will review all the files in `src/` (and potentially subdirectories like `src/contamination/` etc.) for bugs, edge cases, error handling, and style issues, and fix them.
2. **Visualization Suite (`src/attention/visualizer.py`)**:
    - Create a reusable plotting module.
    - `plot_attention_grid(attention_tensor, layer_names=None, head_names=None, ...)`
    - `plot_attention_entropy(entropy_history, ...)`
    - `plot_head_specialization_clusters(attention_tensor, ...)` (UMAP or simple hierarchical clustering for heads)
    - `animate_attention_evolution(attention_history, ...)` (using matplotlib.animation)
3. **Temporal Dynamics Driver (`src/attention/dynamics.py` or script `scripts/plot_dynamics.py`)**:
    - A script/driver to load sequences of checkpoints.
    - Extract attention patterns (using model forwarding with hooks/outputs)
    - Align sequences on absolute steps and grokking pre/post transition phases.
    - Generate the required plots using the visualization suite.
4. **Testing (`tests/test_attention.py`)**:
    - Add robust pytest tests that generate synthetic attention tensors.
    - Assert the shapes and properties of plots (e.g. file existence).
    - Edge cases for clustering (single head, identical heads).
    - Error paths.
5. **Documentation**:
    - Add comprehensive docstrings for all functions and classes in the new modules.
    - Update `README.md` with a new "Gallery" section that shows how to use the visualization suite.
6. **Pre-commit Step**: Verify testing and linting, request instructions.
