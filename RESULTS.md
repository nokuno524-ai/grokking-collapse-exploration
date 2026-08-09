# Grokking and Model Collapse: Findings Summary

This document summarizes the mechanistic analysis and intervention experiments characterizing how synthetic data contamination (model collapse) prevents grokking.

## 1. Mechanistic Analysis

Through our analysis tools we tracked the training dynamics of the modular arithmetic transformer model under different conditions.

- **Attention Circuits (`src/analysis/circuits.py`)**: By evaluating the gradient importance on the `out_proj` matrix over a test grid, we observe the emergence of specific attention heads. In pure training, these heads suddenly spike in importance around the grokking phase transition (step ~1400). When severe collapse is applied, these structures never coalesce.
- **Effective Rank (`src/analysis/weights.py`)**: We compute the Shannon entropy of the normalized singular values for the weight matrices. A sharp reduction in effective rank is observed precisely at the grokking point, representing "representation collapse."
- **Logit Lens (`src/analysis/logit_lens.py`)**: Decoding intermediate states reveals that accurate predictions emerge only at the final MLP layer post-grokking, while attention acts primarily as a routing mechanism.
- **Gradient Noise Scale (`src/train.py`)**: The tracked gradient noise sharply drops off once grokking is achieved as the loss landscape flattens.

## 2. Intervention Experiments

- **Interpolation Threshold (`src/interpolation_study.py`)**: Mixing clean and severely collapsed data shows a critical threshold. A small fraction of collapse (e.g. 5%) delays grokking to step ~3100, while severe collapse (>15%) prevents it entirely within the 50K step window.
- **Recovery (`src/experiments/recovery.py`)**:
    - Weight Resetting: Resetting specific layers and fine-tuning on clean data can recover grokking.
    - LR Annealing: We tested if an annealed schedule can coax a collapsed model out of its local minima.
- **Curriculum Learning (`src/experiments/curriculum.py`)**: Training on collapsed data first, and subsequently switching to clean data, forces the model to unlearn pathological shortcuts and eventually find the generalizing grokking state.

## 3. Publication Figures

The complete set of publication figures (in both PDF and PNG formats) are stored in the `figures/` directory:

1. **Phase Diagram (`figures/phase_diagram.png`)**: Maps the phase transition mapping collapse fraction against the critical grokking step.
2. **Circuit Formation (`figures/circuit_formation.png`)**: Timelines tracking attention head gradient importance across training.
3. **Effective Rank Evolution (`figures/effective_rank.png`)**: Visualizes the geometric compression of the representation spaces (embedding/projections) as grokking occurs.
4. **Gradient Noise (`figures/gradient_noise.png`)**: Tracks the gradient noise scale over training steps.

## Reproducibility

To re-run the full analytical suite, execute:
```bash
./reproduce.sh
```
Use `--quick` for a short test run.
