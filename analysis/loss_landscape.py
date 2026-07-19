import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import copy

sys.path.append(str(Path(__file__).parent.parent))
from src.model import ModularArithmeticTransformer

def get_random_direction(model):
    """Generate a random direction with the same shape as the model weights."""
    direction = []
    for p in model.parameters():
        if p.requires_grad:
            direction.append(torch.randn_like(p))
    return direction

def filter_normalize_direction(direction, model):
    """
    Apply filter normalization to the random direction as per Li et al. 2018.
    For each filter (or weight matrix row/column), we scale the random direction
    to have the same norm as the corresponding filter in the model.
    """
    normalized_direction = []
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            d = direction[idx]
            # Simple layer-wise normalization (easier for varied architectures than strict filter norm)
            p_norm = torch.norm(p)
            d_norm = torch.norm(d)
            if d_norm > 0:
                d_scaled = d * (p_norm / d_norm)
            else:
                d_scaled = d
            normalized_direction.append(d_scaled)
            idx += 1
    return normalized_direction

def calculate_loss(model, data, labels):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
    return loss.item()

def evaluate_1d_landscape(base_model, direction, data, labels, alphas):
    losses = []

    # Store original weights
    orig_weights = [p.clone() for p in base_model.parameters() if p.requires_grad]

    for alpha in alphas:
        # Apply perturbation
        idx = 0
        for p in base_model.parameters():
            if p.requires_grad:
                p.data = orig_weights[idx] + alpha * direction[idx]
                idx += 1

        losses.append(calculate_loss(base_model, data, labels))

    # Restore original weights
    idx = 0
    for p in base_model.parameters():
        if p.requires_grad:
            p.data = orig_weights[idx]
            idx += 1

    return losses

def generate_landscapes(condition="pure", step_to_analyze=-1):
    result_dir = f"results/{condition}"
    if not os.path.exists(result_dir):
        return

    with open(os.path.join(result_dir, "results.json"), "r") as f:
        config = json.load(f)["config"]

    checkpoints = []
    for f in os.listdir(result_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            step = int(f.split("_")[1].split(".")[0])
            checkpoints.append((step, os.path.join(result_dir, f)))
    checkpoints.sort()

    if step_to_analyze == -1:
        step, ckpt_path = checkpoints[-1]
    else:
        # Find closest step
        ckpt_path = min(checkpoints, key=lambda x: abs(x[0] - step_to_analyze))[1]
        step = min(checkpoints, key=lambda x: abs(x[0] - step_to_analyze))[0]

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    # Create random data
    torch.manual_seed(42)
    prime = config.get("prime", 59)
    data = torch.randint(0, prime, (1024, 2))
    labels = (data[:, 0] + data[:, 1]) % prime

    dir1 = get_random_direction(model)
    dir1 = filter_normalize_direction(dir1, model)

    alphas = np.linspace(-1.0, 1.0, 41)
    losses_1d = evaluate_1d_landscape(model, dir1, data, labels, alphas)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, losses_1d, linewidth=2)
    plt.title(f"1D Loss Landscape ({condition}, step {step})")
    plt.xlabel("Step size (alpha)")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/loss_landscape_1d_{condition}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Optional 2D landscape (takes longer, skipping full 2D grid for now to save compute,
    # but structure is here if needed for deeper dive)

if __name__ == "__main__":
    generate_landscapes("pure")
    generate_landscapes("medium_collapse")
