# Contributing to Grokking Collapse

Thank you for your interest in contributing to this project!

## Adding Future Experiments

If you are looking to add new experiments to study grokking and model collapse, please follow these guidelines:

1. **Experimental Settings**:
   - Ensure all hyperparameters, dataset settings, and ablation flags are encapsulated in data classes or dictionaries inside `src/management/config.py` (if it exists) or directly handled without magic numbers.
   - Register your conditions clearly so that `train.py` can invoke them seamlessly via CLI flags.

2. **Data Tracking**:
   - Save your experimental outputs as `results.json` containing the full history trace (test/train accuracies, loss, specific metrics like fourier concentration or diversity).
   - Checkpoints should be saved periodically so that analysis scripts can trace mechanistic evolution over time.

3. **Metrics and Rigor**:
   - Utilize functions from `src/stats_utils.py` for bootstrapping confidence intervals, calculating effect sizes (Cohen's d), and adjusting for multiple comparisons (Bonferroni).
   - This ensures all newly added mechanistic claims are rigorously backed.

4. **Analysis Modules**:
   - **Weight Evolution**: Use `src/weight_analysis.py` to examine layer-wise parameter scaling and singular value transitions.
   - **Attention Visualization**: Use `src/attention_viz.py` for plotting attention evolution, diversity scores, and importance via ablation.
   - **Circuit Discovery**: Use `src/circuit_discovery.py` to perform activation patching and define minimal subgraphs for generalization.

5. **Testing**:
   - Write unit tests for any new analytical function or module in the `tests/` directory.
   - Mock PyTorch models appropriately using the parameters in `src.model.ModularArithmeticTransformer`.
   - Run tests using `PYTHONPATH=. source .venv/bin/activate && pytest tests/`.

6. **Pull Requests**:
   - Describe the purpose of your experiment and the mechanism it intends to clarify.
   - Provide visual artifacts (plots, GIFs) in the PR description whenever feasible.
