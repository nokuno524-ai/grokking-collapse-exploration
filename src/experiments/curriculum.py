import torch
import torch.nn as nn
from pathlib import Path
import json
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.train import train, TrainConfig
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def run_curriculum_experiment(out_dir: str = "results/curriculum", max_steps: int = 50000):
    """
    Train on collapsed data first, then switch to clean data.
    Does grokking still occur? If so, when?
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n--- Curriculum Training ---")
    # Generate collapsed data
    collapsed_config = DatasetConfig(collapse_level=0.5, collapse_severity=0.7)
    c_in, c_tgt, test_in, test_tgt = generate_modular_arithmetic(collapsed_config)
    collapsed_loader = DataLoader(TensorDataset(c_in, c_tgt), batch_size=512, shuffle=True)

    # Generate clean data
    clean_config = DatasetConfig(collapse_level=0.0)
    clean_in, clean_tgt, _, _ = generate_modular_arithmetic(clean_config)
    clean_loader = DataLoader(TensorDataset(clean_in, clean_tgt), batch_size=512, shuffle=True)

    test_loader = DataLoader(TensorDataset(test_in, test_tgt), batch_size=512, shuffle=False)

    model = ModularArithmeticTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

    switch_step = max_steps // 2
    grokked = False
    grokking_step = None

    history = []

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                correct += (logits.argmax(-1) == targets).sum().item()
                total += inputs.size(0)
        return correct / total if total > 0 else 0.0

    c_iter = iter(collapsed_loader)
    cl_iter = iter(clean_loader)

    for step in range(1, max_steps + 1):
        model.train()

        # Use collapsed data for first half, clean for second half
        if step <= switch_step:
            try:
                inputs, targets = next(c_iter)
            except StopIteration:
                c_iter = iter(collapsed_loader)
                inputs, targets = next(c_iter)
        else:
            try:
                inputs, targets = next(cl_iter)
            except StopIteration:
                cl_iter = iter(clean_loader)
                inputs, targets = next(cl_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == max_steps:
            acc = evaluate()
            history.append({"step": step, "test_acc": acc})
            if acc >= 0.95 and not grokked:
                grokked = True
                grokking_step = step
                print(f"Curriculum Grokking achieved at step {step} with acc {acc:.4f}")

            if step % 1000 == 0:
                phase = "Collapsed" if step <= switch_step else "Clean"
                print(f"Step {step} ({phase}): Test Acc = {acc:.4f}")

    results = {
        "switch_step": switch_step,
        "grokked": grokked,
        "grokking_step": grokking_step,
        "history": history
    }

    res_path = out_dir / "curriculum_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Curriculum experiment complete. Results saved to {res_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50000)
    args = parser.parse_args()

    run_curriculum_experiment(max_steps=args.max_steps)
