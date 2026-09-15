## 1. Goal
Fulfill all 6 requirements in the prompt:
1. Extend circuit transplant to cover all severities and both directions.
2. Measure which components carry grokking capability (Feature attribution) by ablating components in the donor.
3. Statistics (seeds >= 3).
4. Code review & fix silent shape/index bugs.
5. Pytest for transplant utilities.
6. Documentation in `RESULTS.md`.

## 2. Approach

**Review and Refactor `transplant_rescue.py` / `head_transplant.py`**:
* The repository memory says "Head-level circuit transplant ... tools are implemented in `src/transplant/head_transplant.py`" and "generalized circuit transplant sweep framework (`src/transplant/sweep.py`) uses a `graft_context`".
* But currently `src/transplant/` only has `transplant_rescue.py` (which I moved). Wait, did I look for `sweep.py`? Let's check memory: "The multi-seed runner for statistical robustness analysis is `src/run_multi_seed_stats.py`". Wait, the memory is from a *future* or *different* state of the repo, or it implies I *should* create these files. The prompt asks me to "extend the circuit transplant machinery". So I will create `src/transplant/sweep.py` and `src/transplant/head_transplant.py` based on the design in memory.

Wait! The Memory specifically mentions:
> Circuit transplantation tools are located in `src/transplant/`. `transplant_rescue.py` handles broad surgical component swaps, while `head_transplant.py` provides a CLI for fine-grained layer-wise, head-wise, bidirectional, and hybrid transplants (with random ablation equivalents) and outputs CSV/Markdown summaries.
> The generalized circuit transplant sweep framework (`src/transplant/sweep.py`) uses a `graft_context` context manager for safe in-place state dict cloning and restoration.

However, a quick `ls` shows `src/transplant/` didn't even exist before I moved `transplant_rescue.py` into it! The memory acts as a set of specifications for what the final architecture *should* look like, or it might be hallucinating files that I need to build. Actually, memory says "Circuit transplantation tools are located in `src/transplant/`" so I should build them there.

**Bug Fixes during Code Review**:
1. Check `random_basis_swap` in `transplant_rescue.py`. The memory mentions: "When implementing random orthonormal basis swaps via SVD (U, S, Vh = torch.linalg.svd(w)), reconstruct the matrix using the transpose of the randomized V-component (e.g., (Ur * S) @ Vr.T) to avoid shape mismatch errors on non-square parameter matrices like embeddings."
Let's check `random_basis_swap`.
It currently does:
```python
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng))
    Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng))
    Vhr = Vr.T
    return Ur @ torch.diag(S) @ Vhr # bug: torch.diag(S) might not match shapes properly if it's not square?
```
For non-square matrices, `U` is `(M, K)`, `S` is `(K,)`, `Vh` is `(K, N)`, where `K = min(M, N)`. `torch.diag(S)` is `(K, K)`. `Ur @ torch.diag(S) @ Vhr` produces `(M, K) @ (K, K) @ (K, N) = (M, N)`, which actually matches shapes! However, memory explicitly says: "reconstruct the matrix using the transpose of the randomized V-component (e.g., `(Ur * S) @ Vr.T`)". We will fix it to `(Ur * S) @ Vhr`.

2. Attention head ablations:
Memory: "To transplant or ablate specific attention heads in PyTorch's `nn.MultiheadAttention`, the fused `in_proj_weight` (shape `3*d_model, d_model`) must be vertically split into Q, K, and V chunks (each `d_model, d_model`), and then sliced by `d_model // n_heads`."
We need to implement this for fine-grained head-level swaps.

3. "When parsing or iterating through run history metrics, explicitly filter out NaN and None values..."
4. "To avoid module shadowing with the src/analysis/ package, the legacy analysis script src/analysis.py was refactored/renamed into src/analysis_legacy.py." (I will do this rename).
5. "When performing zero-shot evaluation or exiting mid-training loops (e.g., in `src/transplant/transplant_rescue.py`), explicitly call `model.eval()` to ensure deterministic evaluations and prevent mode-related leakage." (Fix in `transplant_rescue.py` / new code).

**Feature Attribution & Sweep**:
We need a script to run transplants over all severities (`pure, low, medium, severe, high`), in both directions (`pure->collapsed` and `collapsed->pure`).
Matched training steps: pre-grokking, grokking transition, post-grokking.
Wait, I will need to define these steps.
I'll create a script `src/transplant/sweep.py` to do the multi-seed sweep over the `results/multi_seed/` directory.

**Testing**:
Add tests in `tests/test_transplant.py` for `random_basis_swap`, `patch_state_dict`, and attention head specific surgery.

**Plan**:
1. Fix bugs in `transplant_rescue.py` (`model.eval()`, `random_basis_swap` formula).
2. Rename `src/analysis.py` -> `src/analysis_legacy.py` to avoid package conflicts.
3. Write `src/transplant/head_transplant.py` and `src/transplant/sweep.py` to handle head-level ablations/transplants and the cross-severity matrix.
4. Run the sweep over `results/multi_seed/` for 3+ seeds, testing pure->collapsed and collapsed->pure.
5. Write `tests/test_transplant.py` and ensure they pass.
6. Write the results to `RESULTS.md` and update `README.md`.
