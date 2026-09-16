import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import sys
import os

# Add src to python path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention
from src.analysis.attention.attention_metrics import (
    attention_entropy_per_head,
    head_specialization_score_per_head,
    circuit_formation_index
)
from src.data import DatasetConfig, generate_modular_arithmetic


def load_config(run_dir: Path) -> dict:
    res_path = run_dir / "results.json"
    if res_path.exists():
        with res_path.open() as f:
            return json.load(f).get("config", {})
    return {}

def make_test_loader(cfg: dict, batch_size: int = 512) -> DataLoader:
    dc = DatasetConfig(
        prime=int(cfg.get("prime", 59)),
        train_fraction=float(cfg.get("train_fraction", 0.3)),
        collapse_level=float(cfg.get("collapse_level", 0.0)),
        collapse_severity=float(cfg.get("collapse_severity", 0.5)),
        noise_fraction=float(cfg.get("noise_fraction", 0.0)),
        seed=int(cfg.get("seed", 42)),
    )
    _, _, test_in, test_tgt = generate_modular_arithmetic(dc)
    test_ds = TensorDataset(test_in, test_tgt)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)

def get_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    return ckpts

def main():
    parser = argparse.ArgumentParser(description="Extract attention patterns from a model run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to run directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save extracted attention & metrics")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for test evaluation")
    parser.add_argument("--max-batches", type=int, default=1, help="Number of test batches to extract attention from")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.run_dir)
    if not cfg:
        print(f"Warning: could not load config from {args.run_dir}/results.json. Using defaults.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = make_test_loader(cfg, batch_size=args.batch_size)
    checkpoints = get_checkpoints(args.run_dir)

    if not checkpoints:
        print(f"No checkpoints found in {args.run_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints.")

    metrics_records = []
    prev_attention = None

    model = ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)

    # Pre-fetch evaluation batches
    eval_batches = []
    for i, (x, y) in enumerate(test_loader):
        if i >= args.max_batches:
            break
        eval_batches.append(x.to(device))

    for ckpt_path in checkpoints:
        step = int(re.findall(r"\d+", ckpt_path.name)[-1])
        print(f"Processing step {step}...")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(sd)
        model.eval()

        # We will collect a list of batch attentions for each layer
        # step_attentions[layer_idx] = [batch1_attn, batch2_attn, ...]
        step_attentions = []

        with torch.no_grad():
            for x in eval_batches:
                with extract_attention(model) as extracted:
                    model(x)

                if not step_attentions:
                    # initialize list of lists for each layer
                    step_attentions = [[] for _ in range(len(extracted))]

                for layer_idx, layer_attn in enumerate(extracted):
                    step_attentions[layer_idx].append(layer_attn)

        if not step_attentions:
            continue

        # For each layer, concatenate along batch dimension
        # then we store full_attns as dict mapping layer_idx to np.ndarray
        full_attns = {}
        for layer_idx, layer_batches in enumerate(step_attentions):
            full_attns[layer_idx] = np.concatenate(layer_batches, axis=0)

        # Save raw matrices
        npz_path = args.output_dir / f"attention_step_{step}.npz"
        # store arrays dynamically with their layer index
        np.savez_compressed(npz_path, **{f"layer_{i}": full_attns[i] for i in full_attns})

        if prev_attention is None:
            prev_attention = {}

        for layer_idx, layer_attn in full_attns.items():
            # Compute metrics for this layer
            entropies = attention_entropy_per_head(layer_attn)
            specs = head_specialization_score_per_head(layer_attn)

            cfi = 0.0
            if layer_idx in prev_attention:
                cfi = circuit_formation_index(prev_attention[layer_idx], layer_attn)

            prev_attention[layer_idx] = layer_attn

            record = {
                "step": step,
                "layer": layer_idx,
                "mean_entropy": float(np.mean(entropies)),
                "mean_specialization": float(np.mean(specs)),
                "circuit_formation_index": cfi
            }

            for h in range(len(entropies)):
                record[f"head_{h}_entropy"] = float(entropies[h])
                record[f"head_{h}_specialization"] = float(specs[h])

            metrics_records.append(record)

    if metrics_records:
        csv_path = args.output_dir / "attention_metrics.csv"
        headers = list(metrics_records[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(metrics_records)
        print(f"Saved metrics to {csv_path}")

if __name__ == "__main__":
    main()
