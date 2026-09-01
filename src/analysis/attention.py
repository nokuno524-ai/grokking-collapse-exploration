import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np

class AttentionExtractor:
    """
    Context manager to extract multi-head attention weights from TransformerEncoderLayers.
    Temporarily patches self_attn.forward to intercept kwargs and set need_weights=True.
    Falls back to eager math via SDPA kernel context if necessary.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.maps: Dict[int, torch.Tensor] = {}
        self._original_forwards: Dict[int, Any] = {}

    def __enter__(self):
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers = self.model.transformer.layers
        else:
            raise ValueError("Model does not have 'transformer.layers' attribute")

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'self_attn'):
                continue

            orig_forward = layer.self_attn.forward
            self._original_forwards[i] = orig_forward

            def make_patched_forward(layer_idx, original_fw):
                def patched_forward(query, key, value, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False

                    # Force eager math in PyTorch SDP to guarantee weights are returned
                    if hasattr(torch.nn.attention, 'sdpa_kernel'):
                        with torch.nn.attention.sdpa_kernel(
                            [torch.nn.attention.SDPBackend.MATH]
                        ):
                            attn_out, attn_weights = original_fw(query, key, value, **kwargs)
                    else:
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                            attn_out, attn_weights = original_fw(query, key, value, **kwargs)

                    self.maps[layer_idx] = attn_weights.detach().cpu()
                    return attn_out, attn_weights
                return patched_forward

            layer.self_attn.forward = make_patched_forward(i, orig_forward)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers = self.model.transformer.layers
            for i, layer in enumerate(layers):
                if i in self._original_forwards:
                    layer.self_attn.forward = self._original_forwards[i]


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distributions.
    Args:
        attn_weights: (batch, n_heads, seq_len, seq_len)
    Returns:
        entropy: (batch, n_heads, seq_len)
    """
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    return entropy

def compute_head_specialization_clustering(attn_weights_all_layers: List[torch.Tensor], n_clusters: int = 3, random_state: int = 42) -> np.ndarray:
    """
    Cluster attention heads based on their flattened attention matrices.
    Args:
        attn_weights_all_layers: List of tensors of shape (batch, n_heads, seq_len, seq_len)
            Length of list is n_layers.
    Returns:
        labels: np.ndarray of shape (n_layers * n_heads,)
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("scikit-learn not found, returning dummy clustering")
        num_heads_total = sum([w.shape[1] for w in attn_weights_all_layers])
        return np.zeros(num_heads_total, dtype=int)

    head_features = []
    for w in attn_weights_all_layers:
        batch, n_heads, seq_len, _ = w.shape
        w_permuted = w.permute(1, 0, 2, 3)
        w_flat = w_permuted.reshape(n_heads, -1).numpy()
        head_features.append(w_flat)

    all_heads_features = np.concatenate(head_features, axis=0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(all_heads_features)

    return labels
