1. **Fix `src/contamination_real/mechanistic_metrics.py` (NaN bug)**
   - When fixing the padding issue in `attention_pattern_entropy`, ensure that `Af = Af * k_mask` does not result in zeros that are passed to `.log()`. Use `.clamp_min(1e-12)` again *after* renormalization.
2. **Fix `src/attention/extraction.py` (NaN bug & API issue)**
   - Fix the `compute_entropy` function to clamp probabilities before taking the log to prevent `0.0 * log(0.0) -> NaN`.
   - Remove the hallucinated `torch.backends.mha.get_fastpath_enabled()` calls. PyTorch handles fast-path disabling using the `torch.backends.cuda.sdp_kernel` context manager, which is the proper way, or simply by the fact that we are passing `need_weights=True` which automatically prevents SDPA fast paths.
3. **Verify Fixes**
   - Run the tests again. Add a test in `tests/test_attention.py` to ensure `NaN`s do not occur during `compute_entropy`.
4. **Complete Pre-commit Steps**
   - Run tests, request a final review, initiate memory recording, and submit.
