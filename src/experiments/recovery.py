import torch
import torch.nn as nn
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.train import train, TrainConfig
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def run_recovery_experiment(out_dir: str = "results/recovery", max_steps: int = 5000):
    """
    Test if grokking can be induced in a collapsed model via weight resetting
    or learning rate annealing (LR scheduling).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train a collapsed model (severe) for max_steps
    print("\n--- Training base collapsed model ---")
    base_config = TrainConfig(
        collapse_level=0.5,
        collapse_severity=0.7,
        condition_name="base_collapsed",
        output_dir=str(out_dir),
        max_steps=max_steps
    )
    base_state = train(base_config)

    # Load the base model
    model = ModularArithmeticTransformer().to(device)
    base_ckpt = out_dir / "base_collapsed" / f"checkpoint_{max_steps}.pt"
    if base_ckpt.exists():
        ckpt = torch.load(base_ckpt, map_location=device)
        model_state = {k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}
        model.load_state_dict(model_state)

    # Generate clean data for recovery
    clean_config = DatasetConfig(collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(clean_config)
    train_loader = DataLoader(TensorDataset(train_in, train_tgt), batch_size=512, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_in, test_tgt), batch_size=512, shuffle=False)

    # 2. Reset specific layers (e.g. out_proj) and fine-tune on clean data
    print("\n--- Testing weight reset recovery (clean data) ---")
    # Reset out_proj weight
    nn.init.normal_(model.transformer.layers[0].self_attn.out_proj.weight, std=0.02)
    if model.transformer.layers[0].self_attn.out_proj.bias is not None:
        nn.init.zeros_(model.transformer.layers[0].self_attn.out_proj.bias)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    reset_grokked = False

    # Train for max_steps // 2 to see if it recovers
    for step in range(1, (max_steps // 2) + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            break # Just one batch per step for simplicity here

        if step % 100 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)
                    correct += (logits.argmax(-1) == targets).sum().item()
                    total += inputs.size(0)
            acc = correct / total
            if acc >= 0.95:
                reset_grokked = True
                print(f"Weight Reset Recovered at step {step} with acc {acc:.4f}")
                break

    # 3. LR Annealing (warmup / decay) on collapsed model
    print("\n--- Testing LR Annealing recovery ---")
    # Reload original collapsed model
    if base_ckpt.exists():
        model.load_state_dict(model_state)

    # Use learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps//2)
    lr_grokked = False

    for step in range(1, (max_steps // 2) + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            break

        scheduler.step()

        if step % 100 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)
                    correct += (logits.argmax(-1) == targets).sum().item()
                    total += inputs.size(0)
            acc = correct / total
            if acc >= 0.95:
                lr_grokked = True
                print(f"LR Annealing Recovered at step {step} with acc {acc:.4f}")
                break

    # Save results
    results = {
        "weight_reset_recovered": reset_grokked,
        "lr_annealing_recovered": lr_grokked
    }
    res_path = out_dir / "recovery_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Recovery study complete. Results saved to {res_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50000)
    args = parser.parse_args()

    run_recovery_experiment(max_steps=args.max_steps)
