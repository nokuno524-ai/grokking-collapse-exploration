import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add src to python path for imports if needed
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def get_attention_weights(model, x):
    """
    Extract attention weights from the ModularArithmeticTransformer for input x.

    Args:
        model: ModularArithmeticTransformer instance
        x: Input tensor of shape (batch_size, 2)

    Returns:
        Attention weights tensor of shape (batch_size, n_heads, seq_len, seq_len)
    """
    batch_size = x.shape[0]

    # Token embeddings
    tok = model.token_embed(x)  # (batch, 2, d_model)

    # Positional embeddings
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)  # (batch, 2, d_model)

    # Combine
    h = tok + pos  # (batch, 2, d_model)

    # Extract attention weights manually
    attn_layer = model.transformer.layers[0].self_attn
    qkv = F.linear(h, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    n_heads = model.n_heads
    d_model = model.d_model
    head_dim = d_model // n_heads

    q = q.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)

    import math
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    weights = F.softmax(scores, dim=-1)

    return weights

def compute_attention_entropy(weights):
    """
    Compute entropy of attention weights per head.

    Args:
        weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        entropy: Tensor of shape (batch_size, n_heads, seq_len)
    """
    # weights is a probability distribution over the last dimension (keys)
    # H = -sum(p * log(p))
    epsilon = 1e-10
    entropy = -torch.sum(weights * torch.log(weights + epsilon), dim=-1)
    return entropy

def identify_circuits(weights):
    """
    Identify basic circuit patterns in the attention weights.
    For sequences of length 2, we can identify:
    - Self-attention (attending to the same position)
    - Cross-attention (attending to the other position)

    Args:
        weights: Tensor of shape (batch_size, n_heads, 2, 2)

    Returns:
        metrics: Dictionary with average probabilities for specific patterns per head
    """
    # Assuming seq_len = 2
    batch_size, n_heads, seq_len, _ = weights.shape

    if seq_len != 2:
        raise ValueError("Circuit identification currently only supports seq_len=2")

    # probability of position i attending to position i
    # shape: (batch_size, n_heads, 2)
    self_attn_prob = torch.diagonal(weights, dim1=-2, dim2=-1)

    # mean over batch and sequence positions
    mean_self_attn = self_attn_prob.mean(dim=(0, 2))

    # probability of position i attending to position j (i != j)
    # since sum of probabilities is 1, this is 1 - self_attn_prob
    cross_attn_prob = 1.0 - self_attn_prob
    mean_cross_attn = cross_attn_prob.mean(dim=(0, 2))

    return {
        "self_attention_score": mean_self_attn.cpu().tolist(),
        "cross_attention_score": mean_cross_attn.cpu().tolist()
    }

def analyze_checkpoints(results_dir: str, output_path: str):
    """
    Process all checkpoints across conditions to extract attention patterns.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory {results_dir} not found.")
        return

    print(f"Analyzing checkpoints in {results_dir}...")

    # Store metrics: condition -> list of step metrics
    all_metrics = {}

    # Dataset for evaluating attention (full grid)
    config = DatasetConfig()
    train_in, _, test_in, _ = generate_modular_arithmetic(config)
    full_dataset = torch.cat([train_in, test_in], dim=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = full_dataset.to(device)

    # The expected directory structure is results/<condition>/seed_42/
    conditions = ['pure', 'low_collapse', 'medium_collapse', 'severe_collapse', 'high_collapse']

    for condition in conditions:
        condition_dir = results_path / condition
        if not condition_dir.exists():
            continue

        seed_dirs = [d for d in condition_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
        if not seed_dirs:
            continue

        # Analyze first seed for simplicity in plotting timelines
        seed_dir = seed_dirs[0]

        print(f"Processing {condition}...")
        condition_metrics = []

        # Load run config to match model initialization
        run_config = {}
        results_file = seed_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                run_config = results_data.get('config', {})

        # Find all checkpoints
        checkpoints = list(seed_dir.glob("checkpoint_*.pt"))
        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))

        for ckpt_path in checkpoints:
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                step = ckpt.get('step', int(ckpt_path.stem.split('_')[1]))

                # Initialize model
                model = ModularArithmeticTransformer(
                    prime=run_config.get('prime', 59),
                    d_model=run_config.get('d_model', 128),
                    n_heads=run_config.get('n_heads', 4),
                    d_ff=run_config.get('d_ff', 512),
                    n_layers=run_config.get('n_layers', 1),
                )

                model.load_state_dict(ckpt['model_state'])
                model = model.to(device)
                model.eval()

                with torch.no_grad():
                    # Batch processing to avoid OOM
                    batch_size = 256
                    all_weights = []

                    for i in range(0, len(full_dataset), batch_size):
                        batch = full_dataset[i:i+batch_size]
                        weights = get_attention_weights(model, batch)
                        all_weights.append(weights)

                    # (total_samples, n_heads, seq_len, seq_len)
                    full_weights = torch.cat(all_weights, dim=0)

                    # 1. Attention entropy
                    entropy = compute_attention_entropy(full_weights)
                    # Mean entropy over dataset and sequence length, per head
                    mean_entropy_per_head = entropy.mean(dim=(0, 2)).cpu().tolist()

                    # Mean entropy overall
                    mean_entropy_total = np.mean(mean_entropy_per_head)

                    # 2. Circuit metrics
                    circuits = identify_circuits(full_weights)

                    # 3. Save raw head-to-head metrics for visualization (small sub-sample to save space)
                    # Just keep the mean attention matrix over the dataset
                    mean_attn_matrix = full_weights.mean(dim=0).cpu().tolist()

                metrics = {
                    'step': step,
                    'entropy_per_head': mean_entropy_per_head,
                    'entropy_total': float(mean_entropy_total),
                    'circuits': circuits,
                    'mean_attn_matrix': mean_attn_matrix
                }

                condition_metrics.append(metrics)

            except Exception as e:
                print(f"Error processing {ckpt_path}: {e}")

        all_metrics[condition] = condition_metrics

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    analyze_checkpoints("results", "analysis/attention_metrics.json")
