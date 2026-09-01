import argparse
import glob
from pathlib import Path
import os
import torch
import numpy as np

from src.viz.attention import (
    plot_attention_entropy_over_time,
    plot_head_specialization_heatmap,
    plot_diagnostic_token_traces
)
from src.analysis.attention import (
    compute_attention_entropy,
    compute_head_specialization_clustering
)

def main():
    parser = argparse.ArgumentParser(description="Visualize extracted attention maps.")
    parser.add_argument("--maps-dir", type=str, required=True, help="Directory with .npz files for a run.")
    parser.add_argument("--output-dir", type=str, default="results/viz_attention", help="Output directory for plots.")
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir)
    out_dir = Path(args.output_dir) / maps_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = glob.glob(str(maps_dir / "attn_step_*.npz"))
    if not npz_files:
        print(f"No .npz files found in {maps_dir}")
        return

    # Sort by step
    def extract_step(p):
        return int(Path(p).stem.split('_')[-1])
    npz_files.sort(key=extract_step)

    steps = []
    mean_entropies = []
    traces = {"pos1_to_pos0": [], "pos1_to_pos1": []}

    # For clustering, let's take the final step
    final_maps_dict = None

    n_layers = 0
    n_heads = 0

    for f in npz_files:
        step = extract_step(f)
        steps.append(step)

        data = np.load(f)
        layer_keys = sorted([k for k in data.files if k.startswith("layer_")])
        n_layers = len(layer_keys)

        step_entropies = []
        for l_key in layer_keys:
            attn = torch.tensor(data[l_key])
            if n_heads == 0:
                n_heads = attn.shape[1]

            # Compute entropy
            ent = compute_attention_entropy(attn).mean().item()
            step_entropies.append(ent)

            # Extract trace for diagnostic token (layer 0, head 0 for example)
            # attn shape: (batch, n_heads, seq_len, seq_len)
            # Let's track average attention from pos 1 to pos 0 across batch, layer 0 head 0
            if l_key == "layer_0":
                # batch, n_heads, seq_tgt, seq_src -> [:, 0, 1, 0]
                # Mod arithmetic has seq_len=2 (a, b)
                trace_1_to_0 = attn[:, 0, 1, 0].mean().item()
                trace_1_to_1 = attn[:, 0, 1, 1].mean().item()
                traces["pos1_to_pos0"].append(trace_1_to_0)
                traces["pos1_to_pos1"].append(trace_1_to_1)

        mean_entropies.append(np.mean(step_entropies))
        final_maps_dict = data

    # Plot 1: Entropy over time
    plot_attention_entropy_over_time(
        steps=steps,
        entropies=mean_entropies,
        title=f"Attention Entropy ({maps_dir.name})",
        output_path=out_dir / "entropy.png",
        csv_path=out_dir / "entropy.csv"
    )

    # Plot 2: Traces over time
    plot_diagnostic_token_traces(
        steps=steps,
        traces=traces,
        title=f"L0H0 Diagnostic Traces ({maps_dir.name})",
        output_path=out_dir / "traces.png",
        csv_path=out_dir / "traces.csv"
    )

    # Plot 3: Head clustering at final step
    if final_maps_dict is not None:
        layer_keys = sorted([k for k in final_maps_dict.files if k.startswith("layer_")])
        final_attn_tensors = [torch.tensor(final_maps_dict[k]) for k in layer_keys]

        cluster_labels = compute_head_specialization_clustering(final_attn_tensors, n_clusters=3)
        plot_head_specialization_heatmap(
            cluster_labels=cluster_labels,
            n_layers=n_layers,
            n_heads=n_heads,
            title=f"Head Clusters @ Step {steps[-1]}",
            output_path=out_dir / "clusters.png",
            csv_path=out_dir / "clusters.csv"
        )

    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    main()
