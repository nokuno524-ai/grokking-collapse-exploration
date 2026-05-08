# Grokking Under Distributional Collapse

Can transformer models exhibit grokking (delayed generalization) when trained on data with varying degrees of distributional narrowing (simulating model collapse)?

## Architecture
- Transformer model (214K params) for modular arithmetic
- 5 collapse conditions: Pure, Low (5%), Medium (15%), High (30%), Severe (50%)

## Key Results (CS Cluster, RTX 5080)
- **Pure**: Grokked at step 1400, 100% test acc, Fourier 0.318
- **Low (5%)**: Grokked at step 3100, 97.6% acc, Fourier 0.193
- **Medium (15%)**: NO grokking, 83.9% acc, Fourier 0.170
- **High (30%)**: NO grokking, 31.0% acc, Fourier 0.164
- **Severe (50%)**: NO grokking, 2.7% acc, Fourier 0.114

## Key Finding
Collapse kills grokking above ~10% corruption. 5% doubles grokking time. Fourier spectrum tracks degradation.

## Mechanistic Analysis
- SAE analysis of trained models
- Pure model has 343 special [confidence threshold] features

## Key Paths
- Source: `src/`
- Experiments: `run_experiment.sh`
- Results: `results/`
- Logs: `logs/`
- Setup: `setup.sh`

## GitHub
`nokuno524-ai/grokking-collapse-exploration`
