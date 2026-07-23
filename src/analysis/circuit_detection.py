import torch
import numpy as np
from sklearn.cluster import KMeans

def cluster_attention_patterns(attention_weights: torch.Tensor, n_clusters: int = 3) -> np.ndarray:
    """
    Clusters attention patterns to identify potential algorithmic circuits.
    attention_weights: (batch_size, num_heads, seq_len, seq_len)
    """
    if attention_weights.ndim != 4:
        raise ValueError("attention_weights must be 4D: (batch, heads, seq_len, seq_len)")

    batch_size, num_heads, seq_len, _ = attention_weights.shape

    # Flatten patterns for clustering
    # We cluster over (batch * heads) to find common head patterns across examples
    flattened_patterns = attention_weights.reshape(batch_size * num_heads, seq_len * seq_len).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(flattened_patterns)

    return clusters.reshape(batch_size, num_heads)
