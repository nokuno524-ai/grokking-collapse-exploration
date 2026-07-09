import argparse
import json
import os
import torch
import numpy as np

from src.model import ModularArithmeticTransformer
from src.train import train, TrainConfig

def compute_effective_rank(W: torch.Tensor) -> float:
    """
    Compute effective weight matrix rank via the exponential of the entropy of singular values.
    """
    # SVD
    s = torch.linalg.svdvals(W)
    # Normalize
    s = s / s.sum()
    # Shannon Entropy
    entropy = -(s * torch.log(s + 1e-10)).sum()
    # Effective Rank
    return torch.exp(entropy).item()

def analyze_weights(model: ModularArithmeticTransformer):
    """
    Track SVD of weight matrices and return effective ranks.
    """
    ranks = {}

    # Token Embedding
    W_emb = model.token_embed.weight.detach()
    ranks['token_embed'] = compute_effective_rank(W_emb)

    # Position Embedding
    W_pos = model.pos_embed.weight.detach()
    ranks['pos_embed'] = compute_effective_rank(W_pos)

    # Output Head
    W_out = model.output_head.weight.detach()
    ranks['output_head'] = compute_effective_rank(W_out)

    # Attention layers and FF in TransformerEncoder
    # We have 1 layer
    layer = model.transformer.layers[0]

    # Attention Output projection
    W_attn_out = layer.self_attn.out_proj.weight.detach()
    ranks['attn_out'] = compute_effective_rank(W_attn_out)

    # FFN
    W_ff1 = layer.linear1.weight.detach()
    ranks['ff1'] = compute_effective_rank(W_ff1)

    W_ff2 = layer.linear2.weight.detach()
    ranks['ff2'] = compute_effective_rank(W_ff2)

    return ranks

def main():
    parser = argparse.ArgumentParser(description="Deep Weight Analysis Phase 2")
    parser.add_argument("--collapse_level", type=float, default=0.15)
    parser.add_argument("--noise_rate", type=float, default=0.15)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/phase2_weight_analysis")
    parser.add_argument("--max_steps", type=int, default=10000, help="For testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    for condition in ["pure", "collapse", "noise"]:
        results[condition] = []
        for seed in range(42, 42 + args.num_seeds):
            print(f"--- Weight Analysis: {condition.capitalize()} Seed {seed} ---")

            c_level = args.collapse_level if condition == "collapse" else 0.0
            n_rate = args.noise_rate if condition == "noise" else 0.0

            train_config = TrainConfig(
                prime=59,
                train_fraction=0.3,
                collapse_level=c_level,
                collapse_severity=0.5 if condition == "collapse" else 0.0,
                noise_fraction=n_rate,
                seed=seed,
                output_dir=os.path.join(args.output_dir, f"{condition}_s{seed}"),
                condition_name=condition,
                max_steps=args.max_steps,
                save_every=100,  # Granular tracking for circuit formation
            )

            # Run the training
            state = train(train_config)

            rank_history = []

            for entry in state.history:
                step = entry['step']
                ckpt_path = os.path.join(train_config.output_dir, condition, f"checkpoint_{step}.pt")
                if os.path.exists(ckpt_path):
                    model = ModularArithmeticTransformer(prime=59)
                    try:
                        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    except Exception:
                        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                    model.load_state_dict(checkpoint['model_state'])

                    ranks = analyze_weights(model)
                    ranks['step'] = step
                    rank_history.append(ranks)

            results[condition].append({
                "seed": seed,
                "rank_history": rank_history
            })

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Weight Analysis completed successfully.")

if __name__ == "__main__":
    main()
