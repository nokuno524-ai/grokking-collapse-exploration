"""
Training loop with grokking detection and progress measures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from model import ModularArithmeticTransformer
from data import generate_modular_arithmetic, DatasetConfig, get_all_conditions
from config import TrainConfig, load_config
import metrics


@dataclass
class TrainState:
    """Tracks training state and metrics."""
    step: int = 0
    train_loss: float = float('inf')
    test_loss: float = float('inf')
    train_acc: float = 0.0
    test_acc: float = 0.0
    weight_norm: float = 0.0
    embedding_rank: float = 0.0
    fourier_concentration: float = 0.0
    grokked: bool = False
    grokking_step: Optional[int] = None
    grokking_threshold: float = 0.95
    history: List[dict] = field(default_factory=list)


def set_seed(seed: int):
    """Set all random seeds for perfect reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """Evaluate model, return (loss, accuracy, preds, targets)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * inputs.shape[0]
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += inputs.shape[0]
            all_preds.append(preds)
            all_targets.append(targets)
    
    return total_loss / total, correct / total, torch.cat(all_preds), torch.cat(all_targets)


def train(config: TrainConfig) -> TrainState:
    """Run a single training experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Condition: {config.condition_name}, collapse_level={config.collapse_level}")
    
    # Set seeds
    set_seed(config.seed)
    
    # Generate data
    data_config = DatasetConfig(
        prime=config.prime,
        collapse_level=config.collapse_level,
        collapse_severity=config.collapse_severity,
        seed=config.seed,
    )
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)
    
    train_dataset = TensorDataset(train_in, train_tgt)
    test_dataset = TensorDataset(test_in, test_tgt)

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    # Create model
    model = ModularArithmeticTransformer(
        prime=config.prime,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with weight decay (critical for grokking)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    # Training state
    state = TrainState()
    
    # Output directory
    output_dir = Path(config.output_dir) / config.condition_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    dataloader_iter = iter(train_loader)
    start_time = time.time()
    
    for step in range(1, config.max_steps + 1):
        model.train()
        
        # Get batch
        try:
            inputs, targets = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            inputs, targets = next(dataloader_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state.step = step
        state.train_loss = loss.item()
        
        # Evaluate periodically
        if step % config.eval_every == 0:
            train_loss, train_acc, _, _ = evaluate(model, train_loader, device)
            test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, device)
            
            state.train_loss = train_loss
            state.test_loss = test_loss
            state.train_acc = train_acc
            state.test_acc = test_acc
            state.weight_norm = model.get_weight_norm()
            state.embedding_rank = model.get_embedding_rank()
            state.fourier_concentration = metrics.compute_fourier_concentration(model)

            mode_collapse = metrics.mode_collapse_score(test_preds, config.prime)
            kl_div = metrics.kl_divergence_shift(test_preds, test_targets, config.prime)
            loss_of_complexity = metrics.loss_of_complexity(model)
            memorization = metrics.memorization_score(train_acc, test_acc)
            
            # Detect grokking
            if test_acc >= state.grokking_threshold and not state.grokked:
                state.grokked = True
                state.grokking_step = step
                print(f"🎉 GROKKING at step {step}! Test acc: {test_acc:.4f}")
            
            # Log
            entry = {
                "step": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "weight_norm": state.weight_norm,
                "embedding_rank": state.embedding_rank,
                "fourier_concentration": state.fourier_concentration,
                "mode_collapse": mode_collapse,
                "kl_div": kl_div,
                "loss_of_complexity": loss_of_complexity,
                "memorization": memorization,
            }
            state.history.append(entry)
            
            if step % config.log_every == 0 or state.grokked:
                elapsed = time.time() - start_time
                print(
                    f"Step {step:5d} | "
                    f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                    f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} | "
                    f"‖W‖={state.weight_norm:.2f} rank={state.embedding_rank:.1f} "
                    f"fourier={state.fourier_concentration:.3f} | "
                    f"time={elapsed:.1f}s"
                )
        
        # Save checkpoint
        if step % config.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(config),
            }, ckpt_path)
    
    # Save final results
    results = {
        "config": asdict(config),
        "grokked": state.grokked,
        "grokking_step": state.grokking_step,
        "final_train_acc": state.train_acc,
        "final_test_acc": state.test_acc,
        "final_weight_norm": state.weight_norm,
        "final_embedding_rank": state.embedding_rank,
        "final_fourier_concentration": state.fourier_concentration,
        "history": state.history,
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Grokked: {state.grokked} at step {state.grokking_step}")
    
    return state


def run_all_conditions(output_dir: str = "results", max_steps: int = 50000):
    """Run all experimental conditions."""
    conditions = get_all_conditions()
    results = {}
    
    for name, data_config in conditions.items():
        print(f"\n{'='*60}")
        print(f"Running condition: {name}")
        print(f"{'='*60}")
        
        train_config = TrainConfig(
            collapse_level=data_config.collapse_level,
            collapse_severity=data_config.collapse_severity,
            condition_name=name,
            output_dir=output_dir,
            max_steps=max_steps,
        )
        
        state = train(train_config)
        results[name] = {
            "grokked": state.grokked,
            "grokking_step": state.grokking_step,
            "final_test_acc": state.test_acc,
            "fourier_concentration": state.fourier_concentration,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        status = "✅ GROKKED" if r["grokked"] else "❌ NO GROK"
        print(f"  {name:20s} | {status} | step={r['grokking_step']} | "
              f"test_acc={r['final_test_acc']:.4f} | fourier={r['fourier_concentration']:.3f}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None,
                       help="Run specific condition (pure/low/medium/high/severe)")
    parser.add_argument("--all", action="store_true", help="Run all conditions")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config file")
    args = parser.parse_args()
    
    if args.config:
        train_config = load_config(args.config)
        if args.output_dir != "results":
            train_config.output_dir = args.output_dir
        if args.max_steps != 50000:
            train_config.max_steps = args.max_steps
        train(train_config)
    elif args.all:
        run_all_conditions(args.output_dir, args.max_steps)
    elif args.condition:
        conditions = get_all_conditions()
        # Match partial names
        matched = None
        for name, config in conditions.items():
            if args.condition.lower() in name:
                matched = name
                break
        if matched:
            config = conditions[matched]
            train_config = TrainConfig(
                collapse_level=config.collapse_level,
                collapse_severity=config.collapse_severity,
                condition_name=matched,
                output_dir=args.output_dir,
                max_steps=args.max_steps,
            )
            train(train_config)
        else:
            print(f"Unknown condition: {args.condition}")
            print(f"Available: {list(conditions.keys())}")
    else:
        # Default: run pure condition
        train_config = TrainConfig(condition_name="pure", output_dir=args.output_dir, max_steps=args.max_steps)
        train(train_config)
