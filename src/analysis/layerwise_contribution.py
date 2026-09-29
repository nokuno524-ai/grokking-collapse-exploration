import argparse
from pathlib import Path
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.checkpoint import load_checkpoint, list_checkpoints
from src.transplant.patching import COMPONENT_PATTERNS, get_keys_for_component

logger = logging.getLogger(__name__)

def compute_component_deltas(run_dir: Path, step1: int, step2: int) -> dict:
    """Compute L2 norm difference for each layer/weight matrix between two steps."""
    sd1, _ = load_checkpoint(run_dir, step=step1)
    sd2, _ = load_checkpoint(run_dir, step=step2)

    deltas = {}
    for k in sd1.keys():
        if k in sd2:
            diff = (sd2[k] - sd1[k]).float()
            base = sd1[k].float()
            diff_sq = diff.pow(2).sum().item()
            norm_sq = base.pow(2).sum().item()
            if norm_sq > 0:
                deltas[k] = (diff_sq ** 0.5) / (norm_sq ** 0.5)
            else:
                deltas[k] = 0.0

    return deltas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/layerwise"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for run_dir in args.runs:
        if not run_dir.exists():
            print(f"Skipping {run_dir}, does not exist")
            continue

        res_path = run_dir / "results.json"
        if not res_path.exists():
            continue

        with open(res_path) as f:
            meta = json.load(f)

        grok_step = meta.get("grokking_step")
        if grok_step is None:
            # If not grokked, compare start to end roughly
            ckpts = list_checkpoints(run_dir)
            if len(ckpts) < 2:
                continue
            step1 = ckpts[0][0]
            step2 = ckpts[-1][0]
        else:
            # Compare pre-grok and post-grok
            step1 = max(0, grok_step - 500)
            # fallback logic:
            if "history" in meta and meta["history"]:
                max_step = meta["history"][-1]["step"]
                step2 = min(max_step, grok_step + 500)
            else:
                step2 = grok_step + 500

            ckpts = dict(list_checkpoints(run_dir))

            # Find closest available checkpoints
            available_steps = sorted(ckpts.keys())
            if not available_steps:
                continue

            step1 = min(available_steps, key=lambda x: abs(x - step1))
            step2 = min(available_steps, key=lambda x: abs(x - step2))

        try:
            deltas = compute_component_deltas(run_dir, step1, step2)
            row = {
                "run": str(run_dir),
                "step1": step1,
                "step2": step2,
                "grokked": meta.get("grokked", False)
            }
            row.update(deltas)
            results.append(row)
        except Exception as e:
            print(f"Error computing deltas for {run_dir}: {e}")

    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output_dir / "layerwise_deltas.csv", index=False)

    # Plot heatmap

    # Plot heatmap
    comps = [c for c in df.columns if c not in ["run", "step1", "step2", "grokked"]]
    if comps:
        data = df[comps].values


        plt.figure(figsize=(10, 8))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Relative L2 Change')
        plt.xticks(np.arange(len(comps)), comps, rotation=45, ha='right')
        plt.yticks(np.arange(len(df)), df['run'].apply(lambda x: Path(x).name))
        plt.title('Layerwise Contribution across Transition')
        plt.tight_layout()
        plt.savefig(args.output_dir / "layerwise_heatmap.png")
        plt.close()

if __name__ == "__main__":
    main()
