import os
import argparse
import numpy as np
import glob
from typing import Dict, Tuple
from src.analysis.attention import calculate_attention_entropy, compute_head_similarity, compute_js_divergence
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

def extract_step_from_filename(filename: str) -> int:
    import re
    match = re.search(r"step_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def get_latest_file(directory: str) -> str:
    files = glob.glob(os.path.join(directory, "attn_weights_step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    files_with_steps = [(f, extract_step_from_filename(os.path.basename(f))) for f in files]
    files_with_steps = [fs for fs in files_with_steps if fs[1] >= 0]
    files_with_steps.sort(key=lambda x: x[1])
    return files_with_steps[-1][0]

def cluster_heads(similarity_matrix: np.ndarray, num_clusters: int = 2) -> np.ndarray:
    """Clusters heads based on similarity matrix using hierarchical clustering."""
    # Convert similarity [0,1] to distance [0,1]
    dist_matrix = 1.0 - similarity_matrix
    # Make strictly symmetric and 0 diagonal
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # scipy linkage requires condensed distance matrix
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, 'ward')
    return fcluster(Z, num_clusters, criterion='maxclust')

def main():
    parser = argparse.ArgumentParser(description="Compare attention matrices between pure and collapsed conditions.")
    parser.add_argument("--pure-dir", type=str, required=True, help="Directory with pure condition .npz files.")
    parser.add_argument("--contam-dir", type=str, required=True, help="Directory with contaminated condition .npz files.")
    parser.add_argument("--output", type=str, required=True, help="Path to output markdown file.")

    args = parser.parse_args()

    try:
        pure_file = get_latest_file(args.pure_dir)
        contam_file = get_latest_file(args.contam_dir)
    except FileNotFoundError as e:
        print(e)
        return

    pure_step = extract_step_from_filename(os.path.basename(pure_file))
    contam_step = extract_step_from_filename(os.path.basename(contam_file))

    pure_data = np.load(pure_file)
    contam_data = np.load(contam_file)

    with open(args.output, 'w') as f:
        f.write("# Attention Comparison Report\n\n")
        f.write(f"- **Pure condition step:** {pure_step}\n")
        f.write(f"- **Contaminated condition step:** {contam_step}\n\n")

        # Check common layers
        pure_layers = set(pure_data.files)
        contam_layers = set(contam_data.files)
        common_layers = sorted(list(pure_layers.intersection(contam_layers)))

        if not common_layers:
            f.write("No common layers found between the two conditions.\n")
            return

        for layer_key in common_layers:
            f.write(f"## {layer_key.capitalize()}\n\n")

            pure_attn = pure_data[layer_key]
            contam_attn = contam_data[layer_key]

            # 1. Entropy Deltas
            pure_ent = calculate_attention_entropy(pure_attn).mean(axis=0) # (n_heads,)
            contam_ent = calculate_attention_entropy(contam_attn).mean(axis=0)

            f.write("### Entropy Deltas (Contaminated - Pure)\n")
            f.write("| Head | Pure Entropy | Contam Entropy | Delta |\n")
            f.write("|---|---|---|---|\n")
            for h in range(pure_ent.shape[0]):
                delta = contam_ent[h] - pure_ent[h]
                f.write(f"| {h} | {pure_ent[h]:.3f} | {contam_ent[h]:.3f} | {delta:+.3f} |\n")
            f.write("\n")

            # 2. JS Divergence per head
            f.write("### JS Divergence (Pure vs Contam)\n")
            f.write("| Head | JS Divergence |\n")
            f.write("|---|---|\n")

            # Average over batch to get representative pattern for divergence
            pure_avg = pure_attn.mean(axis=0) # (h, seq, seq)
            contam_avg = contam_attn.mean(axis=0)

            for h in range(pure_attn.shape[1]):
                # Flatten spatial dims and normalize for JSD
                p = pure_avg[h].flatten()
                q = contam_avg[h].flatten()
                p = p / p.sum()
                q = q / q.sum()
                jsd = compute_js_divergence(p, q)
                f.write(f"| {h} | {jsd:.4f} |\n")
            f.write("\n")

            # 3. Cluster Membership Flips
            f.write("### Cluster Membership\n")
            pure_sim = compute_head_similarity(pure_attn)
            contam_sim = compute_head_similarity(contam_attn)

            # Assume 2 clusters (e.g., 'copy' vs 'induction' or 'specialized' vs 'diffuse')
            n_clusters = min(2, pure_attn.shape[1])
            if n_clusters > 1:
                pure_clusters = cluster_heads(pure_sim, n_clusters)
                contam_clusters = cluster_heads(contam_sim, n_clusters)

                f.write("Heads are grouped into 2 clusters based on pattern similarity.\n\n")
                f.write("| Head | Pure Cluster | Contam Cluster | Match? |\n")
                f.write("|---|---|---|---|\n")

                # Align cluster IDs (since clustering is arbitrary, we try to find best alignment)
                # Simple alignment: if cluster 1 in pure mostly maps to cluster 2 in contam, flip contam IDs
                c1_map_c1 = sum(1 for h in range(len(pure_clusters)) if pure_clusters[h] == 1 and contam_clusters[h] == 1)
                c1_map_c2 = sum(1 for h in range(len(pure_clusters)) if pure_clusters[h] == 1 and contam_clusters[h] == 2)

                if c1_map_c2 > c1_map_c1:
                    # Flip contam clusters 1<->2
                    contam_clusters = np.where(contam_clusters == 1, 2, np.where(contam_clusters == 2, 1, contam_clusters))

                for h in range(len(pure_clusters)):
                    match = "Yes" if pure_clusters[h] == contam_clusters[h] else "No"
                    f.write(f"| {h} | {pure_clusters[h]} | {contam_clusters[h]} | {match} |\n")
            else:
                f.write("Only 1 head, clustering not applicable.\n")

            f.write("\n---\n\n")

    print(f"Report generated successfully: {args.output}")

if __name__ == "__main__":
    main()
