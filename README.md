# Grokking Under Distributional Collapse

Can transformer models exhibit grokking (delayed generalization) when trained on data that has undergone varying degrees of distributional narrowing (simulating model collapse)?

## Quick Start

```bash
# Install dependencies
uv venv .venv && source .venv/bin/activate
uv pip install torch numpy matplotlib

# Run a single condition (pure data)
python src/train.py --condition pure --max-steps 50000

# Run all conditions
python src/train.py --all --max-steps 50000

# Analyze results
python src/progress_measures.py results/
python src/analysis.py results/
```

## Experimental Conditions

| Condition | Synthetic Data % | Collapse Severity | Expected Outcome |
|-----------|-----------------|-------------------|------------------|
| pure | 0% | - | Grokking ✅ |
| low_collapse | 5% | 30% | Grokking (delayed) |
| medium_collapse | 15% | 50% | Unstable grokking |
| high_collapse | 30% | 70% | No grokking ❌ |
| severe_collapse | 50% | 90% | No grokking ❌ |

## Key Metrics

- **Test accuracy**: Does the model generalize? (>95% = grokking)
- **Fourier concentration**: Progress measure for circuit formation
- **Embedding rank**: Effective rank of learned representations
- **Weight norm**: Total parameter norm (decreases during grokking)
- **Grokking delay**: Steps between memorization and generalization

## Architecture

- 1-layer transformer, 128 hidden dim, 4 attention heads
- Modular arithmetic task: (a + b) mod 59
- AdamW optimizer with weight decay = 1.0 (critical for grokking)

## References

- Power et al. (2022), "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- Chan et al. (2023), "Progress Measures for Grokking via Mechanistic Interpretability"
- Liu et al. (2022), "Omnigrok: Grokking Beyond Algorithmic Data"
- Shumailov et al. (2023), "The Curse of Recursion: Training on Generated Data Makes Models Forget"
