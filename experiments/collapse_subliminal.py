import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import numpy as np

from src.data import DatasetConfig, generate_modular_arithmetic
from src.model import ModularArithmeticTransformer
from analysis.subliminal_detection import measure_trait_transfer


def track_traits_during_collapse(model: nn.Module, n_traits: int = 5) -> Dict[str, List[float]]:
    """
    Track multiple traits (represented by random directions for now)
    to categorize them as survived, lost, or amplified.
    """
    traits = {
        'survived': [],
        'lost': [],
        'amplified': []
    }

    with torch.no_grad():
        embeddings = model.token_embed.weight.detach() # (prime, d_model)
        for i in range(n_traits):
            direction = torch.randn(model.d_model, device=embeddings.device)
            direction = direction / direction.norm()
            proj = torch.matmul(embeddings, direction).abs().mean().item()

            # Simple thresholding to categorize traits
            if proj > 0.5:
                traits['amplified'].append(proj)
            elif proj > 0.2:
                traits['survived'].append(proj)
            else:
                traits['lost'].append(proj)

    return traits


def train_and_evaluate_collapse_transfer(collapse_severity: float) -> Dict:
    """
    Trains a model on data generated with `collapse_severity` and returns subliminal transfer rates
    along with tracking of specific traits.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure dataset with specific collapse severity
    config = DatasetConfig(
        prime=59,
        train_fraction=0.3,
        collapse_level=0.5, # High collapse level to observe severity effect
        collapse_severity=collapse_severity,
        seed=42
    )

    # Generate data
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)
    train_in, train_tgt = train_in.to(device), train_tgt.to(device)

    # Also generate pure control data
    control_config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, seed=42)
    control_in, _, _, _ = generate_modular_arithmetic(control_config)
    control_in = control_in.to(device)

    # Initialize model
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    # Train for a small number of steps to establish traits
    model.train()
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(train_in)
        loss = criterion(logits, train_tgt)
        loss.backward()
        optimizer.step()

    model.eval()

    # Measure transfer comparing outputs on control vs collapsed inputs
    transfer_score = measure_trait_transfer(control_in, train_in, model)
    traits = track_traits_during_collapse(model)

    return {
        'transfer_rate': transfer_score,
        'traits': traits
    }


def run_collapse_subliminal_experiment(severities: List[float]) -> Dict[float, Dict]:
    """
    Loops through collapse severities and aggregates results.
    """
    results = {}
    for severity in severities:
        print(f"Running experiment for collapse severity: {severity}")
        res = train_and_evaluate_collapse_transfer(severity)
        results[severity] = res
        print(f"Severity {severity} -> Transfer Rate: {res['transfer_rate']:.4f}")
        print(f"  Traits -> Survived: {len(res['traits']['survived'])}, Lost: {len(res['traits']['lost'])}, Amplified: {len(res['traits']['amplified'])}")

    return results


if __name__ == "__main__":
    severities = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = run_collapse_subliminal_experiment(severities)
    print("\nFinal Results:")
    for sev, res in results.items():
        print(f"Severity: {sev:.1f} | Transfer Rate: {res['transfer_rate']:.4f}")
        print(f"  Traits -> Survived: {len(res['traits']['survived'])}, Lost: {len(res['traits']['lost'])}, Amplified: {len(res['traits']['amplified'])}")
