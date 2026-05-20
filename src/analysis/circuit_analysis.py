"""
Circuit-level mechanistic analysis of transformer models.

Tools to decompose attention heads, analyze component importance, and track
weight evolution across phase transitions (grokking vs collapse).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def extract_qkv_weights(
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    d_model: int,
    n_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Q, K, V weights and biases from nn.MultiheadAttention.

    Args:
        in_proj_weight: Tensor of shape (3 * d_model, d_model)
        in_proj_bias: Tensor of shape (3 * d_model)
        d_model: Embedding dimension
        n_heads: Number of attention heads

    Returns:
        (W_q, W_k, W_v, b_q, b_k, b_v) each of shape (d_model, d_model) or (d_model,)
    """
    w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
    return w_q, w_k, w_v, b_q, b_k, b_v


def compute_manual_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    b_q: torch.Tensor,
    b_k: torch.Tensor,
    b_v: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    n_heads: int,
    ablation_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Manual multi-head attention forward pass allowing surgical intervention.

    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        w_q, w_k, w_v: QKV weight matrices (d_model, d_model)
        b_q, b_k, b_v: QKV bias vectors (d_model,)
        out_proj_weight: Output projection weight (d_model, d_model)
        out_proj_bias: Output projection bias (d_model,)
        n_heads: Number of attention heads
        ablation_mask: Optional mask of shape (n_heads,) where 0 means ablate.

    Returns:
        output: Tensor of shape (batch_size, seq_len, d_model)
        attn_probs: Attention probability weights (batch_size, n_heads, seq_len, seq_len)
    """
    batch_size, seq_len, d_model = x.shape
    d_head = d_model // n_heads

    # Compute Q, K, V
    # x is (batch, seq, d_model), w_q is (d_model, d_model)
    q = torch.matmul(x, w_q.t()) + b_q  # (batch, seq, d_model)
    k = torch.matmul(x, w_k.t()) + b_k
    v = torch.matmul(x, w_v.t()) + b_v

    # Reshape for multi-head attention: (batch, n_heads, seq, d_head)
    q = q.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)  # (batch, n_heads, seq, seq)
    attn_probs = F.softmax(scores, dim=-1)

    # Context vector
    context = torch.matmul(attn_probs, v)  # (batch, n_heads, seq, d_head)

    # Ablation intervention
    if ablation_mask is not None:
        # ablation_mask is (n_heads,)
        # Reshape to (1, n_heads, 1, 1) to broadcast
        mask = ablation_mask.view(1, n_heads, 1, 1).to(context.device)
        context = context * mask

    # Reshape back to (batch, seq, d_model)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

    # Output projection
    output = torch.matmul(context, out_proj_weight.t()) + out_proj_bias

    return output, attn_probs


def manual_transformer_forward(
    model: nn.Module,
    x: torch.Tensor,
    ablation_masks: Optional[Dict[int, torch.Tensor]] = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Emulate the ModularArithmeticTransformer forward pass using manual attention computation.
    Supports causal ablations of specific attention heads.

    Args:
        model: ModularArithmeticTransformer instance
        x: Input tensor (batch, 2)
        ablation_masks: Dict mapping layer_idx to (n_heads,) ablation masks.
            For ModularArithmeticTransformer there is only layer_idx 0.

    Returns:
        logits: Output logits of shape (batch, prime)
        all_attn_probs: List of attention probability tensors from each layer.
    """
    batch_size = x.shape[0]

    # Same embedding logic as ModularArithmeticTransformer
    tok = model.token_embed(x)
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    all_attn_probs = []

    # Iterate through transformer layers
    # modular arithmetic model uses nn.TransformerEncoder with nn.TransformerEncoderLayer
    for i, layer in enumerate(model.transformer.layers):
        mask = ablation_masks.get(i) if ablation_masks else None

        # Self Attention block
        # Normalization first? PyTorch default is post-norm unless norm_first=True
        # ModularArithmeticTransformer uses default batch_first=True, norm_first=False

        qkv_weight = layer.self_attn.in_proj_weight
        qkv_bias = layer.self_attn.in_proj_bias
        out_proj_w = layer.self_attn.out_proj.weight
        out_proj_b = layer.self_attn.out_proj.bias

        w_q, w_k, w_v, b_q, b_k, b_v = extract_qkv_weights(
            qkv_weight, qkv_bias, model.d_model, model.n_heads
        )

        attn_out, attn_probs = compute_manual_attention(
            h, w_q, w_k, w_v, b_q, b_k, b_v, out_proj_w, out_proj_b,
            model.n_heads, ablation_mask=mask
        )
        all_attn_probs.append(attn_probs)

        # Residual + Norm 1 (post-norm)
        h = layer.norm1(h + attn_out)

        # FFN block
        ffn_out = layer.linear2(F.gelu(layer.linear1(h)))

        # Residual + Norm 2
        h = layer.norm2(h + ffn_out)

    h = model.ln(h)
    h = h.mean(dim=1)
    logits = model.output_head(h)

    return logits, all_attn_probs


class CircuitDiscoveryTool:
    """
    Tool to measure attention head importance via systematic ablation.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.n_heads = model.n_heads
        self.n_layers = len(model.transformer.layers)

    def get_baseline_performance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute baseline loss on a batch."""
        self.model.eval()
        with torch.no_grad():
            logits, _ = manual_transformer_forward(self.model, x)
            loss = F.cross_entropy(logits, y).item()
        return loss

    def compute_head_importance(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """
        Compute importance score for each head as the increase in loss when the head is ablated.

        Returns:
            importance_scores: ndarray of shape (n_layers, n_heads)
        """
        baseline_loss = self.get_baseline_performance(x, y)
        importance_scores = np.zeros((self.n_layers, self.n_heads))

        self.model.eval()
        with torch.no_grad():
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    mask = torch.ones(self.n_heads, device=self.device)
                    mask[head_idx] = 0.0  # ablate this head

                    logits, _ = manual_transformer_forward(
                        self.model, x, ablation_masks={layer_idx: mask}
                    )
                    ablated_loss = F.cross_entropy(logits, y).item()

                    # Importance = (Ablated Loss - Baseline Loss)
                    importance_scores[layer_idx, head_idx] = ablated_loss - baseline_loss

        return importance_scores


class WeightDecomposition:
    """
    SVD-based weight decomposition to identify grokking vs collapse components.
    """
    @staticmethod
    def get_svd_components(weight_matrix: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top-k SVD components of a weight matrix."""
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            k = min(k, len(S))
            return U[:, :k], S[:k], Vh[:k, :]

    @staticmethod
    def compare_singular_spaces(U1: torch.Tensor, U2: torch.Tensor) -> float:
        """
        Compute the principal angle based overlap between two singular spaces.
        Higher value means spaces are more aligned (max 1.0).
        """
        with torch.no_grad():
            # U1: (D, K), U2: (D, K)
            # Principal angles cosine is the singular values of U1^T U2
            overlap_matrix = torch.matmul(U1.t(), U2)
            S = torch.linalg.svdvals(overlap_matrix)
            # Average cosine of principal angles
            return S.mean().item()


def plot_attention_patterns(
    attn_probs: torch.Tensor,
    title: str = "Attention Patterns",
    out_path: Optional[str] = None
):
    """
    Plot attention probability maps for all heads in a single layer.
    Args:
        attn_probs: Tensor of shape (batch, n_heads, seq_len, seq_len)
    """
    if not HAS_MATPLOTLIB:
        return

    # Average across batch
    avg_probs = attn_probs.mean(dim=0).cpu().numpy()  # (n_heads, seq_len, seq_len)
    n_heads = avg_probs.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i in range(n_heads):
        sns.heatmap(
            avg_probs[i],
            ax=axes[i],
            cmap="Blues",
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            cbar=(i == n_heads - 1)
        )
        axes[i].set_title(f"Head {i}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")

    plt.suptitle(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_head_importance(
    importance_scores: np.ndarray,
    title: str = "Attention Head Importance (Loss Diff)",
    out_path: Optional[str] = None
):
    """
    Plot heatmap of attention head importance scores.
    Args:
        importance_scores: ndarray of shape (n_layers, n_heads)
    """
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        importance_scores,
        cmap="Reds",
        annot=True,
        fmt=".4f",
        xticklabels=[f"Head {i}" for i in range(importance_scores.shape[1])],
        yticklabels=[f"Layer {i}" for i in range(importance_scores.shape[0])]
    )
    plt.title(title)
    plt.xlabel("Attention Heads")
    plt.ylabel("Transformer Layers")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_svd_components(
    singular_values: torch.Tensor,
    title: str = "Singular Value Spectrum",
    out_path: Optional[str] = None
):
    """Plot singular value spectrum of a weight matrix."""
    if not HAS_MATPLOTLIB:
        return

    vals = singular_values.cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(vals) + 1), vals, marker='o')
    plt.title(title)
    plt.xlabel("Component Index")
    plt.ylabel("Singular Value Magnitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.close()
