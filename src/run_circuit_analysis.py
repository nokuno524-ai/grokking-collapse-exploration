"""
Script to apply circuit-level mechanistic analysis on training checkpoints.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig
from src.analysis.circuit_analysis import (
    CircuitDiscoveryTool,
    WeightDecomposition,
    plot_attention_patterns,
    plot_head_importance,
    plot_svd_components,
    manual_transformer_forward
)

def load_checkpoint(ckpt_path: Path) -> ModularArithmeticTransformer:
    """Load model from checkpoint."""
    # Read config implicitly by matching architecture defaults since we lack the raw config dict here
    model = ModularArithmeticTransformer()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    elif "model_state" in state_dict:
        model.load_state_dict(state_dict["model_state"])
    else:
        model.load_state_dict(state_dict)
    return model

def analyze_milestone(
    model: ModularArithmeticTransformer,
    dataset: torch.utils.data.Dataset,
    milestone: int,
    condition: str,
    out_dir: Path
):
    """Run analysis for a single model at a specific milestone."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch of data for analysis
    # Use full validation set equivalent
    X = dataset.tensors[0][:1024]
    Y = dataset.tensors[1][:1024]

    # 1. Circuit Discovery / Head Importance
    tool = CircuitDiscoveryTool(model)
    importance_scores = tool.compute_head_importance(X, Y)
    plot_head_importance(
        importance_scores,
        title=f"Head Importance ({condition}, Step {milestone})",
        out_path=out_dir / f"importance_{condition}_step{milestone}.png"
    )

    # 2. Attention Patterns
    model.eval()
    with torch.no_grad():
        _, attn_probs = manual_transformer_forward(model, X[:32]) # smaller batch for visual clarity
        plot_attention_patterns(
            attn_probs[0], # only layer 0
            title=f"Attention Patterns ({condition}, Step {milestone})",
            out_path=out_dir / f"attn_{condition}_step{milestone}.png"
        )

    # 3. Weight Decomposition (Token Embedding)
    emb_weight = model.token_embed.weight.data
    _, S, _ = WeightDecomposition.get_svd_components(emb_weight, k=30)
    plot_svd_components(
        S,
        title=f"Token Embed SVD ({condition}, Step {milestone})",
        out_path=out_dir / f"svd_{condition}_step{milestone}.png"
    )

def main():
    out_dir = Path("analysis/circuits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define milestones
    milestones = [5000, 15000, 50000]
    conditions = ["pure", "high_collapse"]

    # Generate clean data for evaluation
    # (Analysis should always use clean data to measure true task performance)
    config = DatasetConfig(prime=59)
    X_train, Y_train, X_test, Y_test = generate_modular_arithmetic(config)
    # Combine train and test for full evaluation
    X_full = torch.cat([X_train, X_test])
    Y_full = torch.cat([Y_train, Y_test])
    dataset = torch.utils.data.TensorDataset(X_full, Y_full)

    for condition in conditions:
        for ms in milestones:
            ckpt_path = Path(f"results/{condition}/checkpoint_{ms}.pt")
            if not ckpt_path.exists():
                print(f"Skipping {ckpt_path}, not found.")
                continue

            print(f"Analyzing {condition} at step {ms}...")
            model = load_checkpoint(ckpt_path)
            analyze_milestone(model, dataset, ms, condition, out_dir)

    print("Analysis complete. Visualizations saved to analysis/circuits/")

if __name__ == "__main__":
    main()
