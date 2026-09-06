# Grokking Cliffs — Project Instructions

**Reframed 2026-05-10 after independent audit (`AUDIT_CLAUDE.md`).** The original "model-collapse vs label-noise" framing is empirically refuted by our own baseline. Read `AUDIT_CLAUDE.md` and `RESEARCH_ROADMAP.md` before doing anything new.

## What this project actually shows

Three robust empirical claims (5 seeds each, results in `analysis/exp_c_grid_summary.md`):

1. **Label-noise rate determines a sharp grokking cliff** between noise=0.10 and noise=0.15 at wd ∈ {0.3, 1.0}. wd=3.0 prevents grokking entirely.
2. **Noise ≡ collapse contamination at matched rate.** test_acc and Fourier concentration are statistically indistinguishable at n=5.
3. **Scarcity dissociation.** 50% data still groks, with *higher* Fourier concentration than full data; 15% corrupted data does not grok. Contamination is not effective sample-size shrinkage.

## Architecture (do not change without recording why)

- 1-layer Transformer, d_model=128, 4 heads, d_ff=512, ~214K params.
- Default task `(a+b) mod 59`, train fraction 0.3, AdamW lr=1e-3, batch 512, 50000 steps.
- Default `weight_decay=1.0`. The cliff position is wd-dependent.

## Active experiments (status as of 2026-05-10)

| Exp | Description | Status |
|---|---|---|
| C empirical | 3×6×5 wd × noise × seed grid | done — 90 runs in `results/exp_c_grid/` |
| C theory | Closed-form η*(λ, p, d) + empirical fit | new code added (`src/threshold_theory.py`); needs 1 day analyst time |
| A | Surgical-circuit transplant rescue | new code added (`src/transplant/transplant_rescue.py`); not yet run |
| B | GPT-2 medium + LoRA on contaminated OWT | data-prep died at SLURM time limit; resumable v2 added (`slurm/real_generate_v2.sbatch`) |
| Prime brittleness | p=97 replication of cliff | new code added (`src/run_prime_sweep.py`); cheap, not yet run |

## Hard rules

- No conda. `uv venv .venv && source .venv/bin/activate`.
- `export PYTHONUNBUFFERED=1` in every sbatch.
- Job-name prefix: `grok-` (do not collide with other projects on Rivanna).
- Never modify or cancel jobs that don't start with `grok-`.
- Save checkpoints every 5000 steps — Experiment A reuses them.
- Do **not** add new framing language about "model collapse vs label noise" — that distinction is refuted in our own data and we have moved on.

## Key paths

- Source: `src/`
- SLURM scripts: `slurm/`
- Results (each run is a `results.json` + `checkpoint_*.pt`): `results/`
- Analysis (markdown + plots + CSVs): `analysis/`
- Plan: `NEXT_STAGE.md` (week-by-week, with kill criteria)
- Audit: `AUDIT_CLAUDE.md` (most recent honest read of the data)
- Roadmap: `RESEARCH_ROADMAP.md` (3-experiment salvage plan)

## GitHub

`nokuno524-ai/grokking-collapse-exploration` on `main`. No `gh` CLI on Rivanna; `git push` only.
