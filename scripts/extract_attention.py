import argparse
import os
import glob
from pathlib import Path
import json

import torch
import numpy as np

from src.model import ModularArithmeticTransformer
from src.analysis.attention import AttentionExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract attention maps from a trained model.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing checkpoint_*.pt files.")
    parser.add_argument("--probe-data", type=str, required=True, help="Path to probe inputs .pt file (or generate one).")
    parser.add_argument("--output-dir", type=str, default="analysis/attention_maps", help="Where to save .npz files.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load probe inputs
    if not os.path.exists(args.probe_data):
        print(f"Probe data not found at {args.probe_data}. Creating dummy test batch.")
        # fallback for testing
        x = torch.randint(0, 59, (128, 2))
        torch.save(x, args.probe_data)
    else:
        x = torch.load(args.probe_data, weights_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    # Find checkpoints
    checkpoints = glob.glob(str(ckpt_dir / "checkpoint_*.pt"))
    checkpoints.sort(key=lambda p: int(Path(p).stem.split('_')[1]))

    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    # Extract configs from the first checkpoint to instantiate the model
    first_ckpt = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = first_ckpt.get("config", {})

    # Defaults
    prime = config.get("prime", 59)
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    d_ff = config.get("d_ff", 512)
    n_layers = config.get("n_layers", 1)

    model = ModularArithmeticTransformer(
        prime=prime, d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers
    ).to(device)
    model.eval()

    run_name = ckpt_dir.name
    run_out_dir = out_dir / run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in checkpoints:
        step = int(Path(ckpt_path).stem.split('_')[1])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)

        with AttentionExtractor(model) as extractor:
            with torch.no_grad():
                _ = model(x)

        # extractor.maps contains tensors of shape (batch, n_heads, seq, seq)
        # Convert to numpy and save
        maps_np = {f"layer_{k}": v.numpy() for k, v in extractor.maps.items()}

        out_file = run_out_dir / f"attn_step_{step}.npz"
        np.savez_compressed(out_file, **maps_np)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
