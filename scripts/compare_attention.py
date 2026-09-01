import argparse
import glob
from pathlib import Path
import os
import torch
import numpy as np

from src.analysis.attention import (
    compute_attention_entropy,
    compute_head_specialization_clustering
)

def load_final_maps(maps_dir: Path):
    npz_files = glob.glob(str(maps_dir / "attn_step_*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in {maps_dir}")

    def extract_step(p):
        return int(Path(p).stem.split('_')[-1])
    npz_files.sort(key=extract_step)

    final_file = npz_files[-1]
    step = extract_step(final_file)
    data = np.load(final_file)
    layer_keys = sorted([k for k in data.files if k.startswith("layer_")])
    tensors = [torch.tensor(data[k]) for k in layer_keys]

    return tensors, step

def main():
    parser = argparse.ArgumentParser(description="Compare pure vs collapsed attention maps.")
    parser.add_argument("--pure-dir", type=str, required=True, help="Extracted maps dir for pure run.")
    parser.add_argument("--collapsed-dir", type=str, required=True, help="Extracted maps dir for collapsed run.")
    parser.add_argument("--output-report", type=str, default="analysis/attention_divergence.md", help="Markdown report output path.")
    args = parser.parse_args()

    pure_dir = Path(args.pure_dir)
    col_dir = Path(args.collapsed_dir)

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pure_maps, pure_step = load_final_maps(pure_dir)
        col_maps, col_step = load_final_maps(col_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return

    n_layers = len(pure_maps)
    n_heads = pure_maps[0].shape[1]

    # Compute Entropies
    def get_mean_entropy(maps):
        ents = [compute_attention_entropy(m).mean().item() for m in maps]
        return np.mean(ents)

    pure_ent = get_mean_entropy(pure_maps)
    col_ent = get_mean_entropy(col_maps)

    # Compute Clusters separately
    pure_clusters = compute_head_specialization_clustering(pure_maps, n_clusters=3)
    col_clusters = compute_head_specialization_clustering(col_maps, n_clusters=3)

    # Or jointly to see flips in the same space:
    joint_maps = [torch.cat([pure_maps[i], col_maps[i]], dim=0) for i in range(n_layers)]
    joint_clusters = compute_head_specialization_clustering(joint_maps, n_clusters=3)

    # joint_clusters has shape (n_layers * n_heads,)
    # but the heads are stacked properly if we look at how the function flattens
    # The function permutes batch first into feature dim, so pure and col examples
    # are concatenated along the batch/seq_len dimension within the feature vector.
    # Therefore, each head gets exactly ONE cluster label based on its joint pure+col behavior.
    # Wait, the feature vector is flattened over batch.
    # We want to see if the SAME head behaves differently in pure vs col.
    # So we should cluster pure heads and col heads as separate entities.

    # Better approach:
    # pure heads: n_layers * n_heads feature vectors
    # col heads: n_layers * n_heads feature vectors
    def get_head_features(maps):
        features = []
        for w in maps:
            batch, h, seq, _ = w.shape
            w_perm = w.permute(1, 0, 2, 3)
            features.append(w_perm.reshape(h, -1).numpy())
        return np.concatenate(features, axis=0)

    pure_feats = get_head_features(pure_maps)
    col_feats = get_head_features(col_maps)

    # We must ensure they have the same feature dimension for joint clustering.
    # But batch sizes might differ? If they use the same probe data, batch sizes are equal.
    if pure_feats.shape[1] == col_feats.shape[1]:
        try:
            from sklearn.cluster import KMeans
            all_feats = np.concatenate([pure_feats, col_feats], axis=0)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            all_labels = kmeans.fit_predict(all_feats)
            pure_labels = all_labels[:len(pure_feats)]
            col_labels = all_labels[len(pure_feats):]

            flips = np.sum(pure_labels != col_labels)
        except ImportError:
            pure_labels = pure_clusters
            col_labels = col_clusters
            flips = "N/A (scikit-learn not installed)"
    else:
        pure_labels = pure_clusters
        col_labels = col_clusters
        flips = "N/A (batch size mismatch)"

    report = f"""# Attention Divergence Report

## Experimental Conditions
* **Pure Run:** `{pure_dir.name}` (Step {pure_step})
* **Collapsed Run:** `{col_dir.name}` (Step {col_step})

## Quantitative Divergence

### 1. Mean Attention Entropy
* **Pure:** {pure_ent:.4f} nats
* **Collapsed:** {col_ent:.4f} nats
* **Delta:** {abs(pure_ent - col_ent):.4f} nats

*(A lower entropy indicates a more "spiky" or degenerate attention pattern typical of model collapse.)*

### 2. Head Specialization Clustering
Using K-Means (k=3) on the flattened attention maps across the probe dataset:
* **Total Heads:** {n_layers * n_heads}
* **Cluster Membership Flips:** {flips}

*(A flip indicates that a head changed its fundamental functional role between pure and collapsed conditions.)*
"""

    with open(out_path, "w") as f:
        f.write(report)

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
