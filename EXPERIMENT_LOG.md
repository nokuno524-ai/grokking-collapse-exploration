# Experiment Log

## Goal
To study the interplay between model collapse and grokking in LLMs on a modular arithmetic task.

## Setup
- **Architecture**: 1-layer Transformer encoder, `d_model=128`, `n_heads=4`, `d_ff=512`.
- **Task**: Modular arithmetic `(a + b) mod p`, with `p=59`.
- **Optimization**: AdamW, `lr=1e-3`, `weight_decay=1.0`, `batch_size=512`.

## Runs
- `results/pure`: No data collapse.
- `results/low_collapse`: 5% data replacement, 0.3 severity.
- `results/medium_collapse`: 15% data replacement, 0.5 severity.
- `results/high_collapse`: 30% data replacement, 0.7 severity.
- `results/severe_collapse`: 50% data replacement, 0.9 severity.

## Observations
- **Pure condition** consistently groks at around step 1400, achieving perfect test accuracy.
- **Low collapse condition** delays grokking significantly (around step 3100), reaching ~93% test accuracy.
- **Medium to Severe collapse conditions** never grok. Test accuracy stalls, demonstrating that high contamination prevents the model from generalizing beyond the memorization phase.
- **Weight Norm**: Grokking coincides with a steady weight norm trajectory. In collapsed conditions, the weight norm drops significantly (up to 42% in medium collapse), indicating reduced effective data diversity.

## Next Steps
- Implement causal circuit rescue (Experiment A).
- Verify the exact mathematical relationship derived in Experiment C (Threshold Theory).
