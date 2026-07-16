# Publication Readiness Checklist

This checklist tracks the requirements for submitting this repository as a reproduction package for publication.

## Code & Artifacts
- [x] Full source code is included (`src/`).
- [x] Package dependencies are well defined (can use `uv`).
- [x] Raw data generators are reproducible and verifiable (`tests/test_data_quality.py`).
- [x] Experiment configs are self-contained.

## Reproducibility
- [x] `REPRODUCE.md` provides clear instructions.
- [x] A single script (`run_all.sh`) reproduces all tables and figures.
- [x] Computational requirements (runtime, GPU type, RAM) are documented in scripts and README.
- [x] Results packaging script is included to zip metrics without large models.

## Documentation
- [x] Jupyter notebooks provided for easy interactive verification.
- [x] Core findings are documented in the README.
- [x] Analysis logic (mechanistic interpretability, fourier analysis) documented in code.

## Verification
- [x] Unit tests pass (data quality, script behavior).
- [x] Output structure of the reproducible scripts is validated.
