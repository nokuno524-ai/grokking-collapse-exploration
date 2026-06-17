"""
Curriculum learning experiment to test different data ordering strategies.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import time

try:
    from src.train import TrainState, evaluate, compute_fourier_concentration
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
except ImportError:
    from train import TrainState, evaluate, compute_fourier_concentration
    from model import ModularArithmeticTransformer
    from data import DatasetConfig, generate_modular_arithmetic

@dataclass
class CurriculumConfig:
    strategy: str = "easy_to_hard" # 'easy_to_hard', 'balanced', 'collapse_resistant'
    prime: int = 59
    train_fraction: float = 0.3
    seed: int = 42
    batch_size: int = 512
    max_steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 1.0
    output_dir: str = "results/curriculum"
    condition_name: str = "default"


def generate_curriculum_data(config: CurriculumConfig):
    """
    Generates datasets and a curriculum order of training.
    """
    data_config = DatasetConfig(prime=config.prime, train_fraction=config.train_fraction, seed=config.seed)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)

    if config.strategy == "easy_to_hard":
        # Sort but add a pacing element: mix 70% sorted, 30% random
        # to prevent catastrophic forgetting.
        sort_idx = torch.argsort(train_tgt)
        sorted_in = train_in[sort_idx]
        sorted_tgt = train_tgt[sort_idx]

        # Simple pacing: split into chunks, shuffle within chunks
        chunk_size = len(train_tgt) // 5
        paced_in = []
        paced_tgt = []
        for i in range(5):
            start = i * chunk_size
            end = len(train_tgt) if i == 4 else (i + 1) * chunk_size
            chunk_in = sorted_in[start:end]
            chunk_tgt = sorted_tgt[start:end]

            # Shuffle chunk
            perm = torch.randperm(len(chunk_tgt))
            paced_in.append(chunk_in[perm])
            paced_tgt.append(chunk_tgt[perm])

        train_in = torch.cat(paced_in, dim=0)
        train_tgt = torch.cat(paced_tgt, dim=0)
    elif config.strategy == "balanced":
        # Interleave to ensure uniform distribution of targets
        sort_idx = torch.argsort(train_tgt)
        sorted_in = train_in[sort_idx]
        sorted_tgt = train_tgt[sort_idx]

        # very simple interleaving: take modulo classes sequentially
        interleaved_idx = []
        idx_by_class = [[] for _ in range(config.prime)]
        for i, tgt in enumerate(sorted_tgt.tolist()):
            idx_by_class[tgt].append(i)

        max_len = max(len(lst) for lst in idx_by_class)
        for i in range(max_len):
            for c in range(config.prime):
                if i < len(idx_by_class[c]):
                    interleaved_idx.append(idx_by_class[c][i])

        train_in = sorted_in[interleaved_idx]
        train_tgt = sorted_tgt[interleaved_idx]
    elif config.strategy == "collapse_resistant":
        # Over-sample rare edge cases or uniformize further
        # For simplicity in this experiment, we'll just sort by input sum variance
        a_b_sum = train_in[:, 0] + train_in[:, 1]
        sort_idx = torch.argsort(a_b_sum)
        train_in = train_in[sort_idx]
        train_tgt = train_tgt[sort_idx]

    return train_in, train_tgt, test_in, test_tgt


def run_curriculum_experiment(config: CurriculumConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    train_in, train_tgt, test_in, test_tgt = generate_curriculum_data(config)

    train_dataset = torch.utils.data.TensorDataset(train_in, train_tgt)
    test_dataset = torch.utils.data.TensorDataset(test_in, test_tgt)

    # Do NOT shuffle train_loader since curriculum order matters
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = ModularArithmeticTransformer(prime=config.prime).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    state = TrainState()
    output_dir = Path(config.output_dir) / config.condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader_iter = iter(train_loader)

    for step in range(1, config.max_steps + 1):
        model.train()
        try:
            inputs, targets = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            inputs, targets = next(dataloader_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            train_loss, train_acc = evaluate(model, train_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)

            state.test_acc = test_acc
            state.weight_norm = model.get_weight_norm()
            state.fourier_concentration = compute_fourier_concentration(model)

            if test_acc >= state.grokking_threshold and not state.grokked:
                state.grokked = True
                state.grokking_step = step

            entry = {
                "step": step,
                "test_acc": test_acc,
                "weight_norm": state.weight_norm,
                "fourier_concentration": state.fourier_concentration,
            }
            state.history.append(entry)

    results = {
        "config": asdict(config),
        "grokked": state.grokked,
        "grokking_step": state.grokking_step,
        "final_test_acc": state.test_acc,
        "final_weight_norm": state.weight_norm,
        "history": state.history,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="easy_to_hard")
    args = parser.parse_args()

    config = CurriculumConfig(strategy=args.strategy, condition_name=args.strategy)
    run_curriculum_experiment(config)
