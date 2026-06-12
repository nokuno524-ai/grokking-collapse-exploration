import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add src to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig

matplotlib.use('Agg')

# Styling
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
})

def get_attention_weights(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Extracts attention weights from the model."""
    model.eval()

    # Store attention weights
    attn_weights = []

    def hook(module, input, output):
        # The output of MultiheadAttention in TransformerEncoderLayer is a tuple:
        # (attn_output, attn_output_weights)
        # But we only get the full tuple if we pass need_weights=True to the self_attn layer
        # Since we can't easily change the TransformerEncoderLayer call in ModularArithmeticTransformer
        # without modifying src/model.py, we'll patch the forward call here temporarily.
        pass

    # A more robust way given memory constraints: The memory says:
    # "Attention weights from the ModularArithmeticTransformer can be extracted directly without manual Q/K/V computations by calling the self_attn method of the nn.TransformerEncoderLayer with need_weights=True and average_attn_weights=False."

    # We reproduce the forward pass up to the attention layer.
    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]
    tok = model.token_embed(inputs)
    positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    # In ModularArithmeticTransformer, self.transformer is a TransformerEncoder
    # self.transformer.layers[0] is the TransformerEncoderLayer
    layer = model.transformer.layers[0]

    # The output of self_attn is (attn_output, attn_output_weights)
    # self_attn expects query, key, value
    attn_output, attn_weights = layer.self_attn(
        h, h, h,
        need_weights=True,
        average_attn_weights=False
    )
    return attn_weights.detach().cpu()

def plot_attention_heatmaps(checkpoint_path: str, output_path: str = "visualizations/attention_heatmaps.png"):
    """Plots attention matrices from a given checkpoint on a small batch of test data."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        # Create empty plot
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Checkpoint Found", ha='center', va='center')
        fig.savefig(output_path)
        plt.close(fig)
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    # Init model
    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1),
    )
    # Load weights
    model.load_state_dict(checkpoint["model_state"])

    # Generate a small batch of data
    data_config = DatasetConfig(prime=config.get("prime", 59), seed=42)
    _, _, test_in, _ = generate_modular_arithmetic(data_config)

    # Use just the first few examples
    num_examples = min(4, len(test_in))
    inputs = test_in[:num_examples]

    # Get attention weights: shape (batch, num_heads, seq_len, seq_len)
    attn_weights = get_attention_weights(model, inputs)
    num_heads = attn_weights.shape[1]

    fig, axes = plt.subplots(num_examples, num_heads, figsize=(3*num_heads, 3*num_examples))

    # Handle single example/head cases gracefully
    if num_examples == 1 and num_heads == 1:
        axes = np.array([[axes]])
    elif num_examples == 1:
        axes = np.array([axes])
    elif num_heads == 1:
        axes = np.array([[ax] for ax in axes])

    for i in range(num_examples):
        for h in range(num_heads):
            ax = axes[i, h]
            im = ax.imshow(attn_weights[i, h].numpy(), cmap='viridis', vmin=0, vmax=1)

            # Formatting
            if i == 0:
                ax.set_title(f"Head {h+1}")
            if h == 0:
                a_val, b_val = inputs[i].tolist()
                ax.set_ylabel(f"Example {i+1}\n({a_val} + {b_val})")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['a', 'b'])
            ax.set_yticklabels(['a', 'b'])

            # Add text annotations
            for row in range(2):
                for col in range(2):
                    val = attn_weights[i, h, row, col].item()
                    color = "white" if val < 0.5 else "black"
                    ax.text(col, row, f"{val:.2f}", ha="center", va="center", color=color)

    fig.suptitle(f"Attention Heatmaps (from {os.path.basename(os.path.dirname(checkpoint_path))})", y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved attention heatmaps to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/pure/checkpoint_50000.pt")
    parser.add_argument("--output", type=str, default="visualizations/attention_heatmaps.png")
    args = parser.parse_args()

    plot_attention_heatmaps(args.checkpoint, args.output)
