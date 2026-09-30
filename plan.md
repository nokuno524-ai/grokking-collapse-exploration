1. **Implement Dose-response harness**
   - Edit `src/prediction/dose_response_harness.py` to create `generate_configs(levels, severities, seeds)`.
   - Implement CLI handling `--levels`, `--severities`, `--seeds`, `--dry-run`.
   - Verify with `python src/prediction/dose_response_harness.py --dry-run`.

2. **Implement Early-warning indicators**
   - Edit `src/prediction/early_warning.py`.
   - Implement `compute_effective_rank`, `compute_weight_norm_trajectory`, `compute_attention_entropy_slope`, `compute_train_test_gap_derivative`.
   - Use `uv run -m py_compile src/prediction/early_warning.py` to verify syntax.

3. **Implement Outcome predictor**
   - Edit `src/prediction/outcome_predictor.py` to implement `extract_features` and `train_and_evaluate`.
   - Use `sklearn.linear_model.LogisticRegression` with LOOCV to predict if test accuracy >= 90% using features at 25% of grokking step.
   - Use `uv run -m py_compile src/prediction/outcome_predictor.py` to verify syntax.

4. **Code Review & Fixes**
   - Edit `src/progress_measures.py` using `replace_with_git_merge_diff` to add `if not history: return []` in `compute_learning_speed`.
   - Edit `src/causal_circuit_rescue.py` using `replace_with_git_merge_diff` to ensure that `max_jump` handles arrays with length < 2 or all NaNs properly in `find_sharpest_transition`.
   - Edit `src/leading_indicator_test.py` using `replace_with_git_merge_diff` to ensure `slope_per_1000` returns `None` safely.

5. **Write and run tests**
   - Create `tests/test_prediction.py` and implement unit tests.
   - Run tests with `uv run --with pytest --with scikit-learn pytest tests/`.

6. **Documentation**
   - Ensure all public functions have docstrings.
   - Edit `README.md` to add a "Methods" section.

7. **Pre-commit and Submit**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
   - Run `submit`.
