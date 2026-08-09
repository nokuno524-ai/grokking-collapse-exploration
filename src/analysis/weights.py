import torch
import torch.nn as nn
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.model import ModularArithmeticTransformer

def compute_effective_rank(weight: torch.Tensor) -> float:
    """
    Computes effective rank using Shannon entropy of normalized singular values.
    """
    # SVD of the weight matrix
    # Note: torch.linalg.svdvals is preferred over torch.svd
    # If the weight is a 1D tensor, we can reshape or skip.
    if weight.dim() == 1:
        return 1.0

    s = torch.linalg.svdvals(weight.float())
    s = s / s.sum()

    # Shannon entropy
    entropy = -(s * torch.log(s + 1e-10)).sum()

    return torch.exp(entropy).item()

def track_effective_rank(run_dir: str, prime: int = 59):
    """
    Tracks effective rank of weight matrices over training.
    """
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split('_')[1]))

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)

        step = ckpt["step"]

        # Calculate rank for important matrices
        # Embedding
        emb_rank = compute_effective_rank(model.token_embed.weight.detach())
        # Query/Key/Value/Out projections
        layer = model.transformer.layers[0]
        in_proj_rank = compute_effective_rank(layer.self_attn.in_proj_weight.detach())
        out_proj_rank = compute_effective_rank(layer.self_attn.out_proj.weight.detach())
        # MLPs
        mlp1_rank = compute_effective_rank(layer.linear1.weight.detach())
        mlp2_rank = compute_effective_rank(layer.linear2.weight.detach())
        # Output head
        head_rank = compute_effective_rank(model.output_head.weight.detach())

        res = {
            "step": step,
            "embedding": emb_rank,
            "in_proj": in_proj_rank,
            "out_proj": out_proj_rank,
            "mlp1": mlp1_rank,
            "mlp2": mlp2_rank,
            "output_head": head_rank
        }

        results.append(res)
        print(f"Step {step}: Emb {emb_rank:.1f}, InProj {in_proj_rank:.1f}, OutProj {out_proj_rank:.1f}, MLP1 {mlp1_rank:.1f}, MLP2 {mlp2_rank:.1f}, Head {head_rank:.1f}")

    out_path = run_dir / "effective_rank_tracking.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved rank tracking to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to run directory with checkpoints")
    args = parser.parse_args()

    if args.run_dir:
        track_effective_rank(args.run_dir)
    else:
        print("No run dir provided, doing a dry run...")
        model = ModularArithmeticTransformer()
        emb_rank = compute_effective_rank(model.token_embed.weight.detach())
        layer = model.transformer.layers[0]
        in_proj_rank = compute_effective_rank(layer.self_attn.in_proj_weight.detach())
        print(f"Dry run - Emb rank: {emb_rank:.2f}, In_proj rank: {in_proj_rank:.2f}")
