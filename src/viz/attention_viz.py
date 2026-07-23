import plotly.express as px
import plotly.graph_objects as go
import torch

def plot_attention_heatmap(attention_weights: torch.Tensor, head_idx: int = 0, batch_idx: int = 0):
    """
    Plots a heatmap for a specific attention head and batch example.
    attention_weights: (batch_size, num_heads, seq_len, seq_len)
    """
    pattern = attention_weights[batch_idx, head_idx].detach().cpu().numpy()
    fig = px.imshow(pattern,
                    labels=dict(x="Key Position", y="Query Position", color="Attention"),
                    title=f"Attention Pattern - Head {head_idx}")
    return fig

def plot_head_specialization(attention_weights: torch.Tensor):
    """
    Plots average attention to each position for all heads.
    attention_weights: (batch_size, num_heads, seq_len, seq_len)
    """
    # Average over batch and query positions to see what each head attends to
    avg_attention = attention_weights.mean(dim=(0, 2)).detach().cpu().numpy()

    fig = px.imshow(avg_attention,
                    labels=dict(x="Key Position", y="Head", color="Avg Attention"),
                    title="Head Specialization (Avg Attention to Keys)")
    return fig
