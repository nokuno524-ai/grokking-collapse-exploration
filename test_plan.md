Plan outline:
1. **Dose-response harness**
   - Create `src/prediction/dose_response_harness.py`.
   - Accept inputs to sweep synthetic-data fraction (e.g. `collapse_level`) and mixture temperature (`collapse_severity`) across at least 5 levels (e.g. 0.0, 0.1, 0.2, 0.3, 0.4).
   - Generate configs for training entry points, and support a `--dry-run` mode to only write configs.

2. **Early-warning indicators**
   - In `src/prediction/early_warning.py` or similar, implement calculation of:
     - Effective rank of key weight matrices
     - Weight-norm trajectories
     - Attention-entropy slopes
     - Train-test gap derivative
   - Ensure these are computed from existing checkpoints/run logs.

3. **Outcome predictor**
   - In `src/prediction/outcome_predictor.py`, implement a logistic classifier or threshold classifier to predict grokking success (test accuracy >= 90%).
   - Inputs are the indicators at 25% of the historical cliff step.
   - Use Leave-One-Out Cross-Validation (LOOCV) over available runs.
   - Output AUC and calibration, including on collapsed runs.

4. **Code Review & Fixes**
   - Review `src/prediction` and existing analysis scripts (e.g. `leading_indicator_test.py`, `progress_measures.py`, `causal_circuit_rescue.py`, etc.) for bugs, missing error handling, NaN handling, etc.

5. **Testing**
   - Write `tests/test_prediction.py` utilizing `pytest`.
   - Use synthetic run logs (planted rank collapse, planted entropy drift).
   - Test edge cases and error paths.

6. **Documentation**
   - Ensure docstrings on all public functions/classes in touched/created files.
   - Update README methods section with information tying dose-response curves and early-warning indicators to the collapse-blocks-grokking hypothesis, along with figure captions.
