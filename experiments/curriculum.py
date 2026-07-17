import os
import time
import json
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import asdict

from src.train import TrainConfig, TrainState, compute_fourier_concentration, evaluate
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def get_collapse_level(step: int, max_steps: int, schedule: str, start: float, end: float, step_frac: float) -> float:
    """Determine the collapse level at the current step based on the schedule."""
    if schedule == "constant":
        return end
    elif schedule == "linear":
        progress = step / max_steps
        return start + (end - start) * progress
    elif schedule == "step":
        if step / max_steps < step_frac:
            return start
        else:
            return end
    elif schedule == "reverse":
        # Start collapsed, gradually purify (linear)
        progress = step / max_steps
        return end - (end - start) * progress
    elif schedule == "random":
        # Random schedule between start and end
        import random
        return random.uniform(start, end)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

def train_curriculum(config: TrainConfig, schedule: str, start_col: float, end_col: float, step_frac: float) -> TrainState:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training curriculum on {device}")
    print(f"Condition: {config.condition_name}, schedule={schedule}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # We will generate base dataset configs for 0.0 and 1.0 collapse, or we can just
    # regenerate data dynamically. But generating full data per step is slow.
    # A better approach for continuous curriculum:
    # Generate pure data, and generate fully collapsed data.
    # At each batch, mix them according to the current collapse_level.

    pure_config = DatasetConfig(prime=config.prime, train_fraction=config.train_fraction, collapse_level=0.0, seed=config.seed)
    pure_train_in, pure_train_tgt, test_in, test_tgt = generate_modular_arithmetic(pure_config)

    col_config = DatasetConfig(prime=config.prime, train_fraction=config.train_fraction, collapse_level=1.0, seed=config.seed)
    col_train_in, col_train_tgt, _, _ = generate_modular_arithmetic(col_config)

    test_dataset = TensorDataset(test_in, test_tgt)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = ModularArithmeticTransformer(
        prime=config.prime, d_model=config.d_model, n_heads=config.n_heads, d_ff=config.d_ff, n_layers=config.n_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    state = TrainState()

    output_dir = Path(config.output_dir) / config.condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    n_train = len(pure_train_in)

    # We will just construct batches manually for the curriculum
    for step in range(1, config.max_steps + 1):
        model.train()

        # Determine current collapse level
        current_col = get_collapse_level(step, config.max_steps, schedule, start_col, end_col, step_frac)

        # Sample batch indices
        indices = torch.randint(0, n_train, (config.batch_size,))

        # Decide which samples are collapsed
        is_collapsed = torch.rand(config.batch_size) < current_col

        inputs = torch.where(is_collapsed.unsqueeze(1), col_train_in[indices], pure_train_in[indices])
        targets = torch.where(is_collapsed, col_train_tgt[indices], pure_train_tgt[indices])

        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state.step = step
        state.train_loss = loss.item()

        if step % config.eval_every == 0:
            # For eval, we just evaluate on the current mixed data batch to get a sense of train loss
            # But true train acc should probably be on pure or whatever. We'll use the mixed batch for speed.
            train_loss = loss.item()
            train_acc = (logits.argmax(dim=-1) == targets).float().mean().item()

            test_loss, test_acc = evaluate(model, test_loader, device)

            state.train_loss = train_loss
            state.test_loss = test_loss
            state.train_acc = train_acc
            state.test_acc = test_acc
            state.weight_norm = model.get_weight_norm()
            state.embedding_rank = model.get_embedding_rank()
            state.fourier_concentration = compute_fourier_concentration(model)

            if test_acc >= state.grokking_threshold and not state.grokked:
                state.grokked = True
                state.grokking_step = step
                print(f"🎉 GROKKING at step {step}! Test acc: {test_acc:.4f}")

            entry = {
                "step": step,
                "current_collapse": current_col,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "weight_norm": state.weight_norm,
                "embedding_rank": state.embedding_rank,
                "fourier_concentration": state.fourier_concentration,
            }
            state.history.append(entry)

            if step % config.log_every == 0 or state.grokked:
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | col={current_col:.2f} | "
                      f"test_acc={test_acc:.4f} | ‖W‖={state.weight_norm:.2f}")

    results = {
        "config": asdict(config),
        "schedule": schedule,
        "grokked": state.grokked,
        "grokking_step": state.grokking_step,
        "final_test_acc": state.test_acc,
        "history": state.history,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return state

def run_curriculum_experiment(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    schedules = config.get("schedules", ["linear", "step", "reverse", "random", "constant"])
    start_col = config.get("start_collapse_level", 0.0)
    end_col = config.get("end_collapse_level", 0.5)
    step_frac = config.get("step_transition_frac", 0.5)
    prime = config.get("prime", 59)
    max_steps = config.get("max_steps", 50000)
    eval_every = config.get("eval_every", 100)
    seeds = config.get("seeds", [42, 43, 44])
    output_dir = Path(config.get("output_dir", "results/curriculum"))

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in seeds:
        for schedule in schedules:
            condition_name = f"schedule_{schedule}_seed{seed}"
            train_config = TrainConfig(
                prime=prime,
                max_steps=max_steps,
                eval_every=eval_every,
                seed=seed,
                condition_name=condition_name,
                output_dir=str(output_dir)
            )

            state = train_curriculum(train_config, schedule, start_col, end_col, step_frac)

            all_results.append({
                "schedule": schedule,
                "seed": seed,
                "grokked": state.grokked,
                "grokking_step": state.grokking_step,
                "final_test_acc": state.test_acc
            })

    with open(output_dir / "curriculum_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\\nCurriculum experiments complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/curriculum.yaml")
    args = parser.parse_args()
    run_curriculum_experiment(args.config)
