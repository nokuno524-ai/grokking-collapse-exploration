I need to do ALL of the following:
1. Transplant severity sweep: extend the circuit transplant machinery to cover ALL severity levels (pure, low, medium, severe, high) in both directions (pure→collapsed and collapsed→pure), transplanting at matched training steps (pre-grokking, grokking transition, post-grokking). Emit a transplant success matrix.
2. Feature attribution: for each transplant, measure which components carry grokking capability — attention heads vs MLP layers — by ablating components in the donor before transplant and measuring capability transfer loss. Produce a per-component importance ranking per severity level.
3. Statistical rigor: every transplant cell should be run over >=3 random seeds with mean and std reported; use the repo's existing checkpoint infrastructure.
4. Code Review: review transplant code for silent shape/index bugs (head indexing off-by-one, weight-copy vs reference semantics), device mismatches, and checkpoint schema drift; fix everything found.
5. Testing: pytest for transplant utilities (weight surgery round-trips restore the original function, head-mask application correctness) on tiny CPU models.
6. Documentation: RESULTS.md addendum with the transplant matrix table and interpretation; update README experiment index; docstrings on new analysis functions.
