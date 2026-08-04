import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.model import ModularArithmeticTransformer

CONDITIONS = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

def extract_fourier_features(model):
    """
    Extract Fourier components tracking entropy of the spectrum and peak magnitude.
    """
    # Exclude DC component (0) to measure concentration of non-constant frequencies
    spectrum = model.get_embedding_fourier_spectrum()[1:]
    peak_magnitude = spectrum.max().item()

    # Compute entropy of spectrum
    probs = spectrum / (spectrum.sum(dim=0, keepdim=True) + 1e-10)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    return {
        'peak_magnitude': peak_magnitude,
        'entropy': entropy
    }

def extract_attention_statistics(model, x):
    """
    Extract attention statistics (e.g. entropy of attention weights).
    Since need_weights=False by default in standard nn.TransformerEncoderLayer,
    we compute Q and K projections manually to get the attention distribution.
    """
    batch_size = x.shape[0]

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        emb = tok + pos

        # In ModularArithmeticTransformer, transformer is a TransformerEncoder with 1 layer
        layer = model.transformer.layers[0]
        attn = layer.self_attn

        # Calculate Q, K manually
        d_model = model.d_model

        # in_proj_weight is shape (3 * d_model, d_model)
        # First d_model is Q, second is K, third is V
        q_weight = attn.in_proj_weight[:d_model, :]
        k_weight = attn.in_proj_weight[d_model:2*d_model, :]

        q_bias = attn.in_proj_bias[:d_model] if attn.in_proj_bias is not None else 0
        k_bias = attn.in_proj_bias[d_model:2*d_model] if attn.in_proj_bias is not None else 0

        Q = F.linear(emb, q_weight, q_bias) # (batch, seq_len, d_model)
        K = F.linear(emb, k_weight, k_bias) # (batch, seq_len, d_model)

        # Reshape for multi-head attention
        n_heads = model.n_heads
        head_dim = d_model // n_heads

        Q = Q.view(batch_size, 2, n_heads, head_dim).transpose(1, 2) # (batch, n_heads, seq_len, head_dim)
        K = K.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)

        # Compute scores and weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1) # (batch, n_heads, seq_len, seq_len)

        # Compute entropy of attention weights for each head
        # Average across batch and seq_len (positions)
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean(dim=(0, 2)) # (n_heads,)
        avg_entropy = entropy.mean().item()

    return {
        'avg_entropy': avg_entropy,
        'head_entropies': entropy.tolist()
    }

def track_features(results_dir="results"):
    results_path = Path(results_dir)

    metrics_history = {cond: {'steps': [], 'fourier_entropy': [], 'fourier_peak': [], 'attn_entropy': []} for cond in CONDITIONS}

    x_test = torch.randint(0, 59, (100, 2))

    for condition in CONDITIONS:
        cond_dir = results_path / condition
        if not cond_dir.exists():
            continue

        checkpoints = glob.glob(str(cond_dir / "checkpoint_*.pt"))
        if not checkpoints:
            continue

        def extract_step(ckpt_path):
            filename = os.path.basename(ckpt_path)
            return int(filename.split('_')[1].split('.')[0])

        checkpoints.sort(key=extract_step)

        for ckpt in checkpoints:
            step = extract_step(ckpt)

            model = ModularArithmeticTransformer()
            try:
                state_dict = torch.load(ckpt, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'model_state' in state_dict:
                    state_dict = state_dict['model_state']

                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"Failed to load {ckpt}: {e}")
                continue

            fourier = extract_fourier_features(model)
            try:
                import torch.nn.functional as F
                attn = extract_attention_statistics(model, x_test)
                attn_entropy = attn['avg_entropy']
            except Exception as e:
                print(f"Failed to extract attention: {e}")
                attn_entropy = 0.0

            metrics_history[condition]['steps'].append(step)
            metrics_history[condition]['fourier_entropy'].append(fourier['entropy'])
            metrics_history[condition]['fourier_peak'].append(fourier['peak_magnitude'])
            metrics_history[condition]['attn_entropy'].append(attn_entropy)

    plot_feature_evolution(metrics_history, results_path)

def plot_feature_evolution(metrics_history, results_path):
    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "severe_collapse": "#8e44ad",
        "high_collapse": "#e74c3c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for condition in CONDITIONS:
        data = metrics_history[condition]
        if not data['steps']: continue

        axes[0].plot(data['steps'], data['fourier_entropy'], label=condition, color=colors[condition], linewidth=2)
        axes[1].plot(data['steps'], data['fourier_peak'], label=condition, color=colors[condition], linewidth=2)
        axes[2].plot(data['steps'], data['attn_entropy'], label=condition, color=colors[condition], linewidth=2)

    axes[0].set_title("Fourier Spectrum Entropy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Fourier Peak Magnitude")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Peak Magnitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Attention Entropy")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Entropy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_path / "feature_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved feature evolution plot to results/feature_evolution.png")

if __name__ == "__main__":
    track_features()
