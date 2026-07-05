# Reproducing Grokking and Model Collapse Experiments

This guide explains how to fully reproduce the core findings using the automated experiment runner.

## Quick Start

1.  **Set up the environment:**
    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

2.  **Run all automated experiments:**
    ```bash
    ./run_all.sh
    ```
    This script sequentially runs through `pure`, `low_collapse`, `medium_collapse`, `high_collapse`, and `severe_collapse` conditions. It saves all raw logs in the `logs/` directory and generates checkpoints and JSON metrics in the `results/` directory.

3.  **View the Dashboard:**
    After the script finishes, a `dashboard.html` file will be generated in the root directory. Open it in your web browser to interactively view the multi-panel plots for test accuracy, train loss, weight norm, and Fourier concentration.

## Visualization & Analysis Scripts

After running the experiments, you can further analyze the generated models:

- **Attention Evolution**: Plot the attention heatmaps for key checkpoints using `src/attention_evolution.py`.
- **Weight Analysis**: Analyze effective rank and track Hessian top eigenvalue approximations using `analysis/weight_analysis.py`.
- **Gradient Flow Analysis**: Track gradient norms layer by layer across training using `analysis/gradient_flow.py`.
- **Circuit Tracker**: Identify which attention heads are most active and crucial during grokking transitions by zeroing out their contributions via `analysis/circuit_tracker.py`.
