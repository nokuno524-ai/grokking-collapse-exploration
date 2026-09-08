import os
import argparse
import glob
import numpy as np
import re
from typing import Dict, List, Tuple
from src.viz.attention import plot_attention_heatmaps, plot_entropy_trajectories, plot_head_clustering
from src.analysis.attention import calculate_attention_entropy, compute_head_similarity

def extract_step_from_filename(filename: str) -> int:
    """Extracts step number from filename like 'attn_weights_step_1000.npz'"""
    match = re.search(r"step_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def process_condition_directory(directory: str) -> Tuple[List[int], Dict[str, np.ndarray]]:
    """
    Loads all .npz files in a directory, sorts by step, and calculates entropies.
    Returns:
        steps: List of sorted training steps
        entropies: Dict mapping layer_head name -> array of entropies across steps
    """
    files = glob.glob(os.path.join(directory, "attn_weights_step_*.npz"))
    if not files:
        return [], {}

    # Sort files by step
    files_with_steps = [(f, extract_step_from_filename(os.path.basename(f))) for f in files]
    files_with_steps = [fs for fs in files_with_steps if fs[1] >= 0]
    files_with_steps.sort(key=lambda x: x[1])

    steps = [fs[1] for fs in files_with_steps]

    # Store entropies: { "layer_0_head_0": [val1, val2, ...] }
    entropies = {}

    for f_path, step in files_with_steps:
        data = np.load(f_path)
        for layer_key in data.files:
            layer_idx = int(layer_key.split('_')[1])
            # attention shape: (batch_size, n_heads, seq_len, seq_len)
            attn = data[layer_key]

            # calculate mean entropy over batch for each head
            # entropy shape: (batch_size, n_heads)
            ent = calculate_attention_entropy(attn)
            mean_ent = ent.mean(axis=0)  # shape: (n_heads,)

            for h in range(mean_ent.shape[0]):
                k = f"layer_{layer_idx}_head_{h}"
                if k not in entropies:
                    entropies[k] = []
                entropies[k].append(mean_ent[h])

    # Convert lists to arrays
    for k in entropies:
        entropies[k] = np.array(entropies[k])

    return steps, entropies

def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns and entropies.")
    parser.add_argument("--npz-dirs", type=str, nargs="+", required=True,
                        help="Directories containing .npz files for different conditions.")
    parser.add_argument("--condition-names", type=str, nargs="+", required=True,
                        help="Names of conditions corresponding to --npz-dirs.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save figures.")
    parser.add_argument("--example-step", type=int, default=-1,
                        help="Step to use for heatmaps and clustering. If -1, uses the last step.")

    args = parser.parse_args()

    if len(args.npz_dirs) != len(args.condition_names):
        print("Error: --npz-dirs and --condition-names must have the same number of arguments.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Process all directories to get entropy trajectories
    all_steps = {}
    all_entropies = {}

    for d, name in zip(args.npz_dirs, args.condition_names):
        steps, entropies = process_condition_directory(d)
        if not steps:
            print(f"Warning: No valid .npz files found in {d}")
            continue
        all_steps[name] = steps
        all_entropies[name] = entropies

    if not all_steps:
        print("No data found to plot.")
        return

    # Find common layer/head keys (assume architecture is same across conditions)
    first_cond = list(all_entropies.keys())[0]
    keys = list(all_entropies[first_cond].keys())

    # Group by layer for entropy plots
    layers = set([int(k.split('_')[1]) for k in keys])
    for l_idx in layers:
        # construct dict for plot_entropy_trajectories
        # { condition: { "head_0": array, "head_1": array } }
        cond_data = {}
        for cond_name, ent_dict in all_entropies.items():
            cond_data[cond_name] = {}
            for k, val in ent_dict.items():
                if k.startswith(f"layer_{l_idx}_"):
                    h_idx = int(k.split('_')[3])
                    cond_data[cond_name][f"head_{h_idx}"] = val

        # we assume steps are identical or alignable enough for the first condition
        # (in a real scenario we'd interpolate or align steps strictly, but let's assume matched checkpoints)
        base_steps = all_steps[first_cond]

        out_path = os.path.join(args.output_dir, f"entropy_layer_{l_idx}.png")
        plot_entropy_trajectories(base_steps, cond_data, out_path, layer_idx=l_idx)
        print(f"Saved entropy plot to {out_path}")

    # 2. Plot heatmaps and clustering for a specific step
    for d, name in zip(args.npz_dirs, args.condition_names):
        files = glob.glob(os.path.join(d, "attn_weights_step_*.npz"))
        if not files:
            continue

        files_with_steps = [(f, extract_step_from_filename(os.path.basename(f))) for f in files]
        files_with_steps = [fs for fs in files_with_steps if fs[1] >= 0]
        files_with_steps.sort(key=lambda x: x[1])

        target_file = files_with_steps[-1][0] # Default to last
        if args.example_step != -1:
            exact_matches = [f for f, s in files_with_steps if s == args.example_step]
            if exact_matches:
                target_file = exact_matches[0]
            else:
                print(f"Step {args.example_step} not found in {d}, using last available.")

        data = np.load(target_file)
        target_step = extract_step_from_filename(os.path.basename(target_file))

        # We need a single example for heatmap. Let's take batch index 0.
        for layer_key in data.files:
            l_idx = int(layer_key.split('_')[1])
            attn = data[layer_key]

            # Heatmaps
            out_heatmap = os.path.join(args.output_dir, f"heatmap_{name}_step_{target_step}_layer_{l_idx}.png")
            # attn shape is (batch, heads, seq, seq). We take index 0.
            plot_attention_heatmaps(attn[0], l_idx, out_heatmap, title_suffix=f" ({name}, step {target_step})")
            print(f"Saved heatmap to {out_heatmap}")

            # Clustering (compute similarity matrix for this layer across all batch examples)
            sim_matrix = compute_head_similarity(attn)
            n_heads = attn.shape[1]
            labels = [f"L{l_idx}H{h}" for h in range(n_heads)]
            out_cluster = os.path.join(args.output_dir, f"cluster_{name}_step_{target_step}_layer_{l_idx}.png")
            plot_head_clustering(sim_matrix, labels, out_cluster, title=f"Head Similarity ({name}, Step {target_step})")
            print(f"Saved clustering to {out_cluster}")

if __name__ == "__main__":
    main()
