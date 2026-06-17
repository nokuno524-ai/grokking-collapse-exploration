import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict

from src.data import DatasetConfig, generate_modular_arithmetic
from src.model import ModularArithmeticTransformer


def filter_training_data(
    train_in: torch.Tensor,
    train_tgt: torch.Tensor,
    strategy: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates filtering training data based on different strategies.

    Strategies:
    - 'unfiltered': Return data as is.
    - 'content-filtered': Removes a random subset to simulate semantic filtering.
    - 'persona-aware-filtered': More aggressive targeted filtering.
    """
    n_samples = len(train_tgt)

    if strategy == 'unfiltered':
        return train_in, train_tgt

    elif strategy == 'content-filtered':
        # Simulate simple semantic filtering by dropping 20% of data randomly
        keep_ratio = 0.8
        keep_idx = torch.randperm(n_samples)[:int(n_samples * keep_ratio)]
        return train_in[keep_idx], train_tgt[keep_idx]

    elif strategy == 'persona-aware-filtered':
        # Simulate advanced filtering by dropping 40% of data
        keep_ratio = 0.6
        keep_idx = torch.randperm(n_samples)[:int(n_samples * keep_ratio)]
        return train_in[keep_idx], train_tgt[keep_idx]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_filtering_strategy(collapse_severity: float, strategy: str) -> float:
    """
    Train a model on filtered data and measure the test accuracy (grokking recovery rate).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DatasetConfig(
        prime=59,
        train_fraction=0.3,
        collapse_level=0.3, # Medium collapse level
        collapse_severity=collapse_severity,
        seed=42
    )

    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # Apply filtering
    train_in, train_tgt = filter_training_data(train_in, train_tgt, strategy)

    train_in, train_tgt = train_in.to(device), train_tgt.to(device)
    test_in, test_tgt = test_in.to(device), test_tgt.to(device)

    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    epochs = 150 # Enough to see initial recovery trends
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(train_in)
        loss = criterion(logits, train_tgt)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits = model(test_in)
        preds = test_logits.argmax(dim=-1)
        acc = (preds == test_tgt).float().mean().item()

    return acc


def run_filtering_eval() -> Dict[str, Dict[str, float]]:
    severities = [0.0, 0.5, 0.9]
    strategies = ['unfiltered', 'content-filtered', 'persona-aware-filtered']

    results = {}
    for strategy in strategies:
        results[strategy] = {}
        for severity in severities:
            acc = evaluate_filtering_strategy(severity, strategy)
            results[strategy][f"severity_{severity}"] = acc
            print(f"Strategy: {strategy} | Severity: {severity} | Test Acc: {acc:.4f}")

    return results

if __name__ == "__main__":
    run_filtering_eval()
