import torch
import numpy as np
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from pathlib import Path

# Add the project root to the sys path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viz.attention_evolution import load_attention_weights

def calculate_induction_score(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate the induction score for each head.
    A head is an induction head if it attends from the current token to the previous token.
    For sequence length L=2 (our case), this means attending from position 1 (b) to position 0 (a).

    attn_weights: (n_heads, L, L)
    Returns: (n_heads,) tensor of scores
    """
    n_heads = attn_weights.shape[0]
    scores = torch.zeros(n_heads)

    # Check if sequence length is at least 2
    if attn_weights.shape[1] >= 2 and attn_weights.shape[2] >= 2:
        # Score is the weight of query at pos 1 attending to key at pos 0
        for h in range(n_heads):
            scores[h] = attn_weights[h, 1, 0].item()

    return scores

def cluster_attention_heads(attn_weights: torch.Tensor, n_clusters: int = 2) -> np.ndarray:
    """
    Cluster attention heads based on their flattened attention patterns.

    attn_weights: (n_heads, L, L)
    Returns: Array of cluster labels (n_heads,)
    """
    n_heads = attn_weights.shape[0]

    # Flatten the L x L patterns for each head
    flattened_patterns = attn_weights.reshape(n_heads, -1).numpy()

    # Use KMeans to cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flattened_patterns)

    return labels

def track_circuit_formation(checkpoint_paths: List[str], steps: List[int], dummy_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Track circuit metrics (like induction score) over training time.
    """
    n_steps = len(checkpoint_paths)

    if n_steps == 0:
        return {}

    attn_first = load_attention_weights(checkpoint_paths[0], dummy_input)
    n_heads = attn_first.shape[0]

    metrics = {
        'steps': steps,
        'induction_scores': np.zeros((n_steps, n_heads)),
        'head_clusters': np.zeros((n_steps, n_heads), dtype=int)
    }

    for i, path in enumerate(checkpoint_paths):
        try:
            attn = load_attention_weights(path, dummy_input)

            # Induction scores
            ind_scores = calculate_induction_score(attn)
            metrics['induction_scores'][i] = ind_scores.numpy()

            # Clusters (if we have enough heads to cluster meaningfully)
            if n_heads >= 2:
                n_clusters = min(3, n_heads)
                clusters = cluster_attention_heads(attn, n_clusters=n_clusters)
                metrics['head_clusters'][i] = clusters
        except Exception as e:
            print(f"Error processing {path}: {e}")

    return metrics

def compare_circuit_formation_side_by_side(run_pure_paths: List[str], run_collapsed_paths: List[str], steps: List[int]) -> Dict[str, Any]:
    """
    Compare circuit formation timing between pure and collapsed conditions.
    """
    metrics_pure = track_circuit_formation(run_pure_paths, steps)
    metrics_collapsed = track_circuit_formation(run_collapsed_paths, steps)

    return {
        'steps': steps,
        'pure': metrics_pure,
        'collapsed': metrics_collapsed
    }

if __name__ == "__main__":
    # Test script with dummy checkpoint
    checkpoint_path = "tests/data/dummy_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print("Testing circuit discovery analysis...")
        attn = load_attention_weights(checkpoint_path)

        ind_scores = calculate_induction_score(attn)
        print(f"Induction scores for {attn.shape[0]} heads: {ind_scores.numpy()}")

        if attn.shape[0] >= 2:
            clusters = cluster_attention_heads(attn)
            print(f"Head clusters: {clusters}")

        metrics = track_circuit_formation([checkpoint_path, checkpoint_path], [0, 1000])
        print("Tracked metrics shape:")
        print(f"  induction_scores: {metrics['induction_scores'].shape}")
        print(f"  head_clusters: {metrics['head_clusters'].shape}")
    else:
        print("Run tests/generate_checkpoint.py first to create a dummy checkpoint.")