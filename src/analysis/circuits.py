import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json

# Adjust imports according to repo structure
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def compute_head_importance(model: ModularArithmeticTransformer, dataloader, device):
    """
    Computes importance of each attention head via gradient norm on out_proj.weight.
    """
    model.eval()

    # Enable gradients for parameters
    for param in model.parameters():
        param.requires_grad = True

    head_importances = torch.zeros(model.n_heads, device=device)
    total_loss = 0

    # We only need one batch or a few batches to get gradients
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        # In ModularArithmeticTransformer, transformer uses nn.TransformerEncoder
        # The first layer is model.transformer.layers[0]
        # its self_attn has out_proj
        # We need to get gradients of out_proj.weight
        layer = model.transformer.layers[0]
        out_proj_grad = layer.self_attn.out_proj.weight.grad # (d_model, d_model)

        if out_proj_grad is not None:
            # The out_proj weight has shape (d_model, d_model)
            # where the input is concatenated head outputs.
            # We can compute importance by slicing the gradient by head dimension.
            head_dim = model.d_model // model.n_heads
            for h in range(model.n_heads):
                start_idx = h * head_dim
                end_idx = start_idx + head_dim
                # The gradient w.r.t the h-th head's input to out_proj
                head_grad = out_proj_grad[:, start_idx:end_idx]
                head_importances[h] += head_grad.abs().mean().item()

        total_loss += loss.item()
        break # One batch is usually enough for gradient-based importance tracking on a deterministic dataset

    return head_importances / len(dataloader)


def track_circuits_over_training(run_dir: str, prime: int = 59):
    """
    Tracks attention circuits over training steps for a given run directory.
    """
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split('_')[1]))

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    config = DatasetConfig(prime=prime)
    _, _, test_in, test_tgt = generate_modular_arithmetic(config)
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(test_in, test_tgt)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    results = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_config = ckpt.get("config", {})

        model = ModularArithmeticTransformer(
            prime=model_config.get("prime", prime),
            d_model=model_config.get("d_model", 128),
            n_heads=model_config.get("n_heads", 4),
            d_ff=model_config.get("d_ff", 512),
            n_layers=model_config.get("n_layers", 1),
        ).to(device)

        model_state = ckpt["model_state"]
        # Remove module prefix if exists
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)

        step = ckpt["step"]
        importances = compute_head_importance(model, test_loader, device)

        results.append({
            "step": step,
            "head_importances": importances.tolist()
        })

        print(f"Step {step}: {importances.tolist()}")

    out_path = run_dir / "circuit_tracking.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved circuit tracking to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to run directory with checkpoints")
    args = parser.parse_args()

    if args.run_dir:
        track_circuits_over_training(args.run_dir)
    else:
        print("No run dir provided, doing a dry run...")
        model = ModularArithmeticTransformer()
        device = torch.device("cpu")
        config = DatasetConfig()
        _, _, test_in, test_tgt = generate_modular_arithmetic(config)
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(test_in, test_tgt)
        test_loader = DataLoader(test_dataset, batch_size=32)
        importances = compute_head_importance(model, test_loader, device)
        print("Dry run importances:", importances)
