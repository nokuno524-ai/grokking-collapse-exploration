with open("README.md", "r") as f:
    code = f.read()
code = code.replace(
    '3. **Outcome Predictor (`src/prediction/outcome_predictor.py`)**: A logistic regression classifier that takes these early signals and predicts final grokking success via Leave-One-Out Cross-Validation (LOOCV). It reliably predicts grokking failures on collapsed datasets long before they actually stall.',
    '3. **Outcome Predictor (`src/prediction/outcome_predictor.py`)**: A logistic regression classifier that takes these early signals and predicts final grokking success via Leave-One-Out Cross-Validation (LOOCV). It reliably predicts grokking failures on collapsed datasets long before they actually stall.\n\n### Figure Interpretations\n- **Dose-Response Heatmaps**: These plots map `collapse_level` vs `collapse_severity`, illuminating the sharp, precise cliff beyond which grokking becomes structurally blocked.\n- **Early-Warning Trajectories**: Time series plots contrasting collapsed vs. pure runs at step ranges far earlier than grokking, visualizing diverging attention entropy, weight norms, and rank collapse before test accuracy gives any signal.'
)
with open("README.md", "w") as f:
    f.write(code)
