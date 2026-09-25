import argparse
import os
import json
from pathlib import Path
import torch
import numpy as np

from src.model import ModularArithmeticTransformer
from src.attention.extraction import extract_attention
from src.attention.visualization import plot_attention_diff

def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Generate an attention diff view between two checkpoints.")
    parser.add_argument("--ckpt-a", type=str, required=True, help="Path to checkpoint A (e.g. donor).")
    parser.add_argument("--ckpt-b", type=str, required=True, help="Path to checkpoint B (e.g. collapsed).")
    parser.add_argument("--prompt", type=int, nargs=2, default=[7, 13], help="Two integers representing the prompt (a, b).")
    parser.add_argument("--output-dir", type=str, default="analysis/attention_vis", help="Directory to save the diff view.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a = load_model(args.ckpt_a, device)
    model_b = load_model(args.ckpt_b, device)

    x = torch.tensor([args.prompt], dtype=torch.long, device=device)

    with extract_attention(model_a) as ext_a:
        with torch.no_grad():
            model_a(x)

    with extract_attention(model_b) as ext_b:
        with torch.no_grad():
            model_b(x)

    if not ext_a or not ext_b:
        print("Failed to extract attention from one or both models.")
        return

    attn_a = ext_a[0][0][0]
    attn_b = ext_b[0][0][0]

    tokens = [str(args.prompt[0]), str(args.prompt[1])]
    save_path = output_dir / f"attention_diff_{args.prompt[0]}_{args.prompt[1]}.png"

    plot_attention_diff(
        attn_a=attn_a,
        attn_b=attn_b,
        tokens=tokens,
        layer_idx=0,
        save_path=str(save_path),
        title_a="Model A",
        title_b="Model B"
    )

    sidecar = {
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "prompt": args.prompt
    }
    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Diff visualization saved to {save_path}")

if __name__ == "__main__":
    main()
