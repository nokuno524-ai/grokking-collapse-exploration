"""
Training loop with grokking detection and progress measures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    # Import as a package: `from src.train import train`
    from .model import ModularArithmeticTransformer
    from .data import generate_modular_arithmetic, DatasetConfig, get_all_conditions
except ImportError:
    # Run as a script: `python src/train.py`
    from model import ModularArithmeticTransformer
    from data import generate_modular_arithmetic, DatasetConfig, get_all_conditions


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    prime: int = 59
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 1
    
    # Training
    max_steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 1.0  # Key hyperparameter for grokking!
    batch_size: int = 512
    
    # Logging
    eval_every: int = 100
    log_every: int = 50
    save_every: int = 5000
    
    # Data
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    train_fraction: float = 0.3
    noise_fraction: float = 0.0
    seed: int = 42
    
    # Output
    output_dir: str = "results"
    condition_name: str = "default"


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


def compute_fourier_concentration(model: ModularArithmeticTransformer, top_k: int = 5) -> float:
    """
    Measure how concentrated the Fourier spectrum is on the top-k frequencies.
    High concentration → grokking has occurred (or is occurring).

    Args:
        model (ModularArithmeticTransformer): The model whose embedding to analyze.
        top_k (int): Number of top frequencies to measure concentration against.

    Returns:
        float: The ratio of energy in the top-k frequencies relative to the total energy.
    """
    spectrum = model.get_embedding_fourier_spectrum()  # (prime, d_model)
    # Average across embedding dimensions
    avg_spectrum = spectrum.mean(dim=1)  # (prime,)
    # Exclude DC component
    avg_spectrum = avg_spectrum[1:]
    total_energy = avg_spectrum.sum()
    if total_energy < 1e-10:
        return 0.0
    top_energy = avg_spectrum.topk(min(top_k, len(avg_spectrum))).values.sum()
    return (top_energy / total_energy).item()


def load_checkpoint(ckpt_path: str) -> dict:
    """
    Load a PyTorch checkpoint robustly with fallback for older versions.

    Args:
        ckpt_path (str): Filepath to the PyTorch checkpoint.

    Returns:
        dict: The loaded checkpoint dictionary containing state_dict and configs.
    """
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Failed to load with weights_only=True: {e}. Falling back to weights_only=False.")
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """
    Evaluate model, return (loss, accuracy).

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): PyTorch DataLoader providing the evaluation batches.
        device (torch.device): Device to perform evaluation on.

    Returns:
        tuple: (average_loss, average_accuracy) across all batches.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
            total_loss += float(loss.item()) * inputs.shape[0]
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += inputs.shape[0]
    
    return total_loss / total, correct / total


def train(config: TrainConfig) -> TrainState:
    """
    Run a single training experiment loop, evaluating and tracking metrics periodically.

    Args:
        config (TrainConfig): Training configuration parameters.

    Returns:
        TrainState: Object containing the final tracked state and history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Condition: {config.condition_name}, collapse_level={config.collapse_level}")
    
    # Set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Generate data
    data_config = DatasetConfig(
        prime=config.prime,
        train_fraction=config.train_fraction,
        collapse_level=config.collapse_level,
        collapse_severity=config.collapse_severity,
        noise_fraction=config.noise_fraction,
        seed=config.seed,
    )
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)
    
    train_dataset = TensorDataset(train_in, train_tgt)
    test_dataset = TensorDataset(test_in, test_tgt)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=loader_generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
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
            train_loss, train_acc = evaluate(model, train_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            
            state.train_loss = train_loss
            state.test_loss = test_loss
            state.train_acc = train_acc
            state.test_acc = test_acc
            state.weight_norm = model.get_weight_norm()
            state.embedding_rank = model.get_embedding_rank()
            state.fourier_concentration = compute_fourier_concentration(model)
            
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
    """
    Run all predefined experimental conditions for collapse severity mapping.

    Args:
        output_dir (str): Directory to save outputs to.
        max_steps (int): Maximum training steps per condition.

    Returns:
        dict: A dictionary of results summary per condition.
    """
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Convert hydra config to our TrainConfig dataclass
    config = TrainConfig(
        prime=cfg.prime,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        max_steps=cfg.max_steps,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        eval_every=cfg.eval_every,
        log_every=cfg.log_every,
        save_every=cfg.save_every,
        collapse_level=cfg.collapse_level,
        collapse_severity=cfg.collapse_severity,
        train_fraction=cfg.train_fraction,
        noise_fraction=cfg.noise_fraction,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        condition_name=cfg.condition_name,
    )
    # Check if a meta-condition was requested via run_all config logic
    # Here we simplify, letting the user pass override pairs
    # e.g., python src/train.py collapse_level=0.15 condition_name=medium_collapse
    train(config)

if __name__ == "__main__":
    main()
