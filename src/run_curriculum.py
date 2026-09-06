"""
Curriculum Rescue Runner
Starts training on fully-collapsed data, then at a configurable step S switches to fresh data.
Logs test accuracy, train loss, and weight norms through the transition.
"""

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Wrap local imports in try...except for script execution
try:
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.model import ModularArithmeticTransformer
    from src.train import TrainState, compute_fourier_concentration, evaluate
    from src.curriculum.schedules import StepPhaseOutSchedule, LinearDecaySchedule
    from src.curriculum.mixer import DataMixer
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.model import ModularArithmeticTransformer
    from src.train import TrainState, compute_fourier_concentration, evaluate
    from src.curriculum.schedules import StepPhaseOutSchedule, LinearDecaySchedule
    from src.curriculum.mixer import DataMixer


@dataclass
class CurriculumConfig:
    """Configuration for curriculum experiments."""
    prime: int = 59
    train_fraction: float = 0.3
    collapse_severity: float = 0.9  # Severe collapse
    tree_depth: int = 1

    # Schedule params
    schedule_type: str = "step"  # "step" or "linear"
    switch_step: int = 20000     # For step schedule
    linear_end_ratio: float = 1.0 # For linear schedule
    start_fresh: float = 0.0     # Initial fresh fraction
    end_fresh: float = 1.0       # Final fresh fraction

    # Model params
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_ff: int = 512

    # Training params
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1.0
    max_steps: int = 50000
    seed: int = 42

    # Log params
    eval_every: int = 100
    log_every: int = 1000
    save_every: int = 10000
    output_dir: str = "results/curriculum_rescue"
    run_name: str = "rescue"


def run_curriculum(config: CurriculumConfig) -> TrainState:
    """
    Run a curriculum training experiment.

    Starts training on a collapsed distribution, then mixes with a fresh distribution
    according to a specified schedule. Tracks training metrics and weight norms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Run: {config.run_name}, Schedule: {config.schedule_type}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # 1. Generate Fresh Dataset
    fresh_config = DatasetConfig(
        prime=config.prime, train_fraction=config.train_fraction,
        collapse_level=0.0, seed=config.seed
    )
    fresh_train_in, fresh_train_tgt, test_in, test_tgt = generate_modular_arithmetic(fresh_config)

    # 2. Generate Collapsed Dataset
    col_config = DatasetConfig(
        prime=config.prime, train_fraction=config.train_fraction,
        collapse_level=1.0, collapse_severity=config.collapse_severity,
        tree_depth=config.tree_depth, seed=config.seed
    )
    col_train_in, col_train_tgt, _, _ = generate_modular_arithmetic(col_config)

    test_dataset = TensorDataset(test_in, test_tgt)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 3. Setup Schedule and Mixer
    if config.schedule_type == "step":
        schedule = StepPhaseOutSchedule(
            switch_step=config.switch_step,
            before_fresh=config.start_fresh,
            after_fresh=config.end_fresh
        )
    elif config.schedule_type == "linear":
        schedule = LinearDecaySchedule(
            start_fresh=config.start_fresh,
            end_fresh=config.end_fresh,
            end_step_ratio=config.linear_end_ratio
        )
    else:
        raise ValueError(f"Unknown schedule type: {config.schedule_type}")

    mixer = DataMixer(
        fresh_inputs=fresh_train_in, fresh_targets=fresh_train_tgt,
        collapsed_inputs=col_train_in, collapsed_targets=col_train_tgt,
        schedule=schedule, batch_size=config.batch_size, seed=config.seed
    )

    # 4. Model and Optimizer
    model = ModularArithmeticTransformer(
        prime=config.prime, d_model=config.d_model, n_heads=config.n_heads,
        d_ff=config.d_ff, n_layers=config.n_layers
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    state = TrainState()

    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for step in range(1, config.max_steps + 1):
        model.train()

        inputs, targets = mixer.get_batch(step, config.max_steps)
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state.step = step
        state.train_loss = loss.item()

        if step % config.eval_every == 0:
            # We evaluate on fresh train set for train_loss/train_acc to measure true performance
            # (evaluating on the mixed batch is too noisy)
            model.eval()
            with torch.no_grad():
                # Take a subset of fresh data for quick eval
                eval_idx = torch.randperm(len(fresh_train_in))[:config.batch_size]
                eval_in = fresh_train_in[eval_idx].to(device)
                eval_tgt = fresh_train_tgt[eval_idx].to(device)
                eval_logits = model(eval_in)
                train_loss = F.cross_entropy(eval_logits, eval_tgt).item()
                train_acc = (eval_logits.argmax(dim=-1) == eval_tgt).float().mean().item()

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
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "weight_norm": state.weight_norm,
                "embedding_rank": state.embedding_rank,
                "fourier_concentration": state.fourier_concentration,
                "fresh_fraction": schedule.get_fresh_fraction(step, config.max_steps)
            }
            state.history.append(entry)

            if step % config.log_every == 0 or state.grokked:
                elapsed = time.time() - start_time
                print(
                    f"Step {step:5d} [{schedule.get_fresh_fraction(step, config.max_steps):.2f} fresh] | "
                    f"test_acc={test_acc:.4f} | ‖W‖={state.weight_norm:.2f} | time={elapsed:.1f}s"
                )

        if step % config.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(config),
            }, ckpt_path)

    # Save results
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

    print(f"\nResults saved to {results_path}")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run curriculum rescue experiments")
    parser.add_argument("--sweep-step", action="store_true", help="Sweep over switch_step")
    parser.add_argument("--switch-step", type=int, default=20000)
    parser.add_argument("--schedule", type=str, default="step", choices=["step", "linear"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/curriculum_rescue")
    parser.add_argument("--max-steps", type=int, default=50000)
    args = parser.parse_args()

    if args.sweep_step:
        steps = [5000, 10000, 15000, 20000, 25000, 30000]
        for s in steps:
            print(f"\n{'='*60}")
            print(f"Running rescue with switch_step = {s}")
            print(f"{'='*60}")
            cfg = CurriculumConfig(
                schedule_type="step",
                switch_step=s,
                seed=args.seed,
                output_dir=args.output_dir,
                run_name=f"switch_{s}_seed_{args.seed}",
                max_steps=args.max_steps
            )
            run_curriculum(cfg)
    else:
        cfg = CurriculumConfig(
            schedule_type=args.schedule,
            switch_step=args.switch_step,
            seed=args.seed,
            output_dir=args.output_dir,
            run_name=f"{args.schedule}_{args.switch_step}_seed_{args.seed}",
            max_steps=args.max_steps
        )
        run_curriculum(cfg)
