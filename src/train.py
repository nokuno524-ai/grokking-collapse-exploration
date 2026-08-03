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
    
    # Task
    task: str = "modular_arithmetic"

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


def compute_hessian_max_eigenvalue(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, max_iters: int = 10) -> float:
    """Computes the largest eigenvalue of the Hessian matrix using power iteration."""
    # Disable SDP backends on CPU to avoid RuntimeError as instructed in AGENTS.md / memory
    if not inputs.is_cuda:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    model.eval()

    # We only care about parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    # First forward and backward to get gradients
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)

    # Compute first-order gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    # Initialize random vector v with the same shape as parameters
    v = [torch.randn_like(p) for p in params]

    # Normalize v
    v_norm = torch.sqrt(sum(torch.sum(x**2) for x in v))
    v = [x / v_norm for x in v]

    eigenvalue = 0.0

    for _ in range(max_iters):
        # Compute dot product of gradients and v
        grad_v = sum(torch.sum(g * x) for g, x in zip(grads, v))

        # Compute Hessian-vector product
        # H*v = d(grad_v)/d(params)
        Hv = torch.autograd.grad(grad_v, params, retain_graph=True)

        # Compute Rayleigh quotient: v^T * H * v
        eigenvalue = sum(torch.sum(x * y) for x, y in zip(v, Hv)).item()

        # Update v for next iteration
        Hv_norm = torch.sqrt(sum(torch.sum(x**2) for x in Hv))
        if Hv_norm == 0:
            break
        v = [x / Hv_norm for x in Hv]

    return eigenvalue

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, compute_info: bool = False) -> tuple:
    """Evaluate model, return (loss, accuracy) and optionally information metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_inputs = []
    all_logits = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * inputs.shape[0]
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += inputs.shape[0]

            if compute_info and len(all_inputs) < 10: # limit info computation to some batches to save memory
                all_inputs.append(inputs)
                all_logits.append(logits)

    if total == 0:
        return 0.0, 0.0, {} if compute_info else (0.0, 0.0)

    avg_loss = total_loss / total
    acc = correct / total
    
    if compute_info and len(all_inputs) > 0:
        from analysis.information import compute_information_flow
        cat_inputs = torch.cat(all_inputs, dim=0)
        cat_logits = torch.cat(all_logits, dim=0)
        info_metrics = compute_information_flow(model, cat_inputs, cat_logits)
        return avg_loss, acc, info_metrics

    if compute_info:
        return avg_loss, acc, {}

    return avg_loss, acc


def train(config: TrainConfig) -> TrainState:
    """Run a single training experiment."""
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
        task=config.task,
    )

    if config.task == "modular_arithmetic":
        from data import generate_modular_arithmetic
        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)
    elif config.task == "group_multiplication":
        from data import generate_group_multiplication
        train_in, train_tgt, test_in, test_tgt = generate_group_multiplication(data_config)
    elif config.task == "binary_addition":
        from data import generate_binary_addition
        train_in, train_tgt, test_in, test_tgt = generate_binary_addition(data_config)
    elif config.task == "sparse_parity":
        from data import generate_sparse_parity
        train_in, train_tgt, test_in, test_tgt = generate_sparse_parity(data_config)
    elif config.task == "in_context_learning":
        from data import generate_in_context_learning
        train_in, train_tgt, test_in, test_tgt = generate_in_context_learning(data_config)
    else:
        raise ValueError(f"Unknown task: {config.task}")
    
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

        # Compute gradient norm
        grad_norm = 0.0
        layer_grad_norms = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                g_norm = p.grad.norm().item()
                grad_norm += g_norm ** 2
                layer_grad_norms[f"grad_norm_{name}"] = g_norm
        grad_norm = grad_norm ** 0.5

        optimizer.step()
        
        state.step = step
        state.train_loss = loss.item()
        
        # Evaluate periodically
        if step % config.eval_every == 0:
            # compute Hessian max eigenvalue on a subset of training data
            hessian_max_ev = compute_hessian_max_eigenvalue(model, inputs[:64], targets[:64])
            
            # Compute gradient noise scale on a couple batches
            model.eval()
            grad_variances = []
            tmp_grads = []
            n_batches_for_noise = 5
            dl_iter = iter(train_loader)
            with torch.enable_grad():
                for _ in range(n_batches_for_noise):
                    try:
                        b_in, b_tgt = next(dl_iter)
                    except StopIteration:
                        break
                    b_in, b_tgt = b_in.to(device), b_tgt.to(device)
                    optimizer.zero_grad()
                    b_logits = model(b_in)
                    b_loss = F.cross_entropy(b_logits, b_tgt)
                    b_loss.backward()

                    batch_grads = []
                    for p in model.parameters():
                        if p.grad is not None:
                            batch_grads.append(p.grad.detach().clone().flatten())

                    if batch_grads:
                        tmp_grads.append(torch.cat(batch_grads))

            if len(tmp_grads) > 1:
                stacked_grads = torch.stack(tmp_grads) # (n_batches, total_params)
                grad_noise_scale = stacked_grads.var(dim=0).mean().item()
            else:
                grad_noise_scale = 0.0

            model.train() # restore train mode
            optimizer.zero_grad() # clear these tmp grads
            train_loss, train_acc = evaluate(model, train_loader, device)
            # Evaluate with compute_info on test set
            eval_res = evaluate(model, test_loader, device, compute_info=True)
            if len(eval_res) == 3:
                test_loss, test_acc, info_metrics = eval_res
            else:
                test_loss, test_acc = eval_res
                info_metrics = {}

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
                "grad_norm": grad_norm,
                "grad_noise_scale": grad_noise_scale,
                "hessian_max_eigenvalue": hessian_max_ev,
            }
            entry.update(layer_grad_norms)
            # Add info metrics if available
            entry.update(info_metrics)

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


def run_all_conditions(output_dir: str = "results", max_steps: int = 50000) -> dict:
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
    args = parser.parse_args()
    
    if args.all:
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
        train_config = TrainConfig(condition_name="pure", output_dir=args.output_dir)
        train(train_config)
