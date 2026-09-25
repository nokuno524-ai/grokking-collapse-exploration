1. **Code Review & Fixes**
   - Review `src/model.py`: Ensure `ModularArithmeticTransformer` correctly implements the attention mechanism. Verify `attention_mask` passing to `transformer` if necessary (though for (a+b) mod p it's fixed length 2, no padding mask needed). Add a parameter to `nn.TransformerEncoderLayer` or patch it to capture attention weights if possible (or use hooks).
   - Review checkpoint loading: The user mentions "checkpoint-loading utilities for bugs, style issues, error handling gaps, and edge cases (missing checkpoints, config mismatch, attention-mask handling)". `src/transplant_rescue.py` has `load_run` which is fragile (uses regex list index). I'll create `src/checkpoint.py` with robust `load_run` handling strict config checking and `model.eval()`.

2. **Attention Extraction Module**
   - According to memory: "Attention extraction is implemented via the `extract_attention` context manager in `src/attention_extraction.py` (which also houses the `compute_entropy` helper). It temporarily patches `nn.MultiheadAttention.forward` using `*args, **kwargs` to intercept calls, and explicitly disables PyTorch 2.0+ SDPA fast-paths using `torch.backends.mha.set_fastpath_enabled(False)` during `eval()` mode to ensure hooks fire."
   - Create `src/attention_extraction.py` that implements a context manager that patches `MultiheadAttention.forward` to yield attention matrices.
   - It will run a fixed evaluation batch, extract per-layer per-head attention matrices with metadata (step, regime, seed, sample class), and cache to `.npz`.

3. **Visualization Scripts**
   - Create `scripts/vis_attention.py` that generates:
     - (a) per-head attention heatmaps (start/mid/grok-point/end)
     - (b) step-indexed animations or small-multiple grids
     - (c) summary metrics per head over time (entropy, diagonal dominance, off-diagonal mass) with seed variability bands.
     - Saves PNGs and CSVs.

4. **Testing**
   - Create `tests/test_attention.py` testing extraction on a tiny transformer.
   - Test hook registration, caching, metric computation, and error paths.

5. **Documentation**
   - Add docstrings to all public functions in the new files.
   - Update `README.md` with visualization workflow.

6. **Pre-commit Checks**
   - Call `pre_commit_instructions` tool and perform testing, verification, review.
