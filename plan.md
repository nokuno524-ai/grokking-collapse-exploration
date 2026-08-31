1. **Analyze codebase & Plan architecture:**
   - I moved `src/transplant_rescue.py` to `src/transplant/run_transplants.py` as it seems to be the intended home for this. Wait, the prompt says `Prior sessions built circuit-transplant experiments...` and `Memory` says `Circuit transplantation tools for swapping... are located in src/transplant/circuits.py... The experiment matrix harness for evaluating these transplants is src/transplant/run_transplants.py`. Let me verify `src/transplant/circuits.py` - it might be missing or I might have overwritten something.
   - Actually, I just created the `src/transplant` directory and put `run_transplants.py` there. The `Memory` is describing the *ideal* state or maybe a previous state. I should check if I missed any files. I should refactor `src/transplant/run_transplants.py` to expose a programmatically callable function instead of just a script, so I can reuse it in `replication_harness.py`.
   - `src/transplant/stats.py`: Write functions for Cohen's d (or similar effect size metric), bootstrap confidence intervals, and a function to check replication across seeds (sign and magnitude).
   - `src/transplant/replication_harness.py`: Iterate over multiple seeds (e.g. 42, 43, 44, 45, 46). For each seed, run the transplant script for conditions (pure vs pure, pure vs low, pure vs medium/severe). Aggregate the VariantResults across seeds. Compute stats using `stats.py`. Generate the consolidated JSON and markdown tables.
   - Plot script (optional/guarded): `src/transplant/plot_replication.py` for effect-size forest plots per condition.
   - Verdict document: `transplant_replication.md` inside `analysis/` or `docs/`.
   - Code Review & Bug fixes: Check seed threading, verify checkpoints, metric aggregation, add type hints.
   - Testing: `tests/test_transplant_stats.py` for testing stats functions and a smoke test of the replication harness on CPU (fast).

2. **Step 1: Write `src/transplant/stats.py` and its tests**
   - Implement bootstrap CI: sample with replacement, compute mean, repeat N times, return percentiles (e.g. 2.5%, 97.5%).
   - Implement Cohen's d: (mean(group1) - mean(group2)) / pooled_stdev. Handling zero variance, n=1.
   - Implement cross-seed replication check: Does effect size maintain the same sign across all seeds?
   - Test them in `tests/test_transplant_stats.py`.

3. **Step 2: Refactor `run_transplants.py` to be callable programmatically**
   - Extract the core logic of `main()` into `run_transplant_experiment(pure_run_dir, contam_run_dir, components, output_dir, ...)`.
   - Ensure it works with both CPU and GPU.

4. **Step 3: Write `src/transplant/replication_harness.py`**
   - Load seeds (e.g., from `results/multi_seed/`).
   - Iterate over conditions (low_collapse, medium_collapse, etc.).
   - For each condition and seed, call `run_transplant_experiment(pure_run_dir=results/multi_seed/{seed}/pure, contam_run_dir=results/multi_seed/{seed}/{condition})`.
   - Collect all results (train acc, test acc, etc.).
   - Calculate stats (Cohen's d, CI, cross-seed check) comparing baseline_contam to transplant_<C> for each component.
   - Save to JSON and Markdown.

5. **Step 4: Create plotting script and Document findings**
   - `src/transplant/plot_replication.py`.
   - `analysis/transplant_replication.md`.
