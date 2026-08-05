import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import ModularArithmeticTransformer

def get_attention_weights(model, x):
    # Manually compute Q, K, V
    # x shape: (batch, 2)
    tok = model.token_embed(x)
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    pos = model.pos_embed(positions)
    h = tok + pos # (batch, 2, d_model)

    # Layer 0 TransformerEncoderLayer
    layer = model.transformer.layers[0]

    # Pre-attention layer norm (if batch_first=True, norm1 is applied)
    h_norm = layer.norm1(h)

    # Extract projection weights from MultiheadAttention
    in_proj_weight = layer.self_attn.in_proj_weight
    in_proj_bias = layer.self_attn.in_proj_bias

    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    # Project
    qkv = F.linear(h_norm, in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape for heads
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2) # (batch, heads, seq, head_dim)
    k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

    # Scaled dot product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1) # (batch, heads, seq, seq)

    return attn_weights

def compute_attention_entropy(attn_weights):
    # attn_weights: (batch, heads, seq, seq)
    # Entropy = - sum(p * log(p))
    # Add epsilon to prevent log(0)
    eps = 1e-10
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    return entropy.mean().item()

def analyze_checkpoints(base_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    conditions = ["pure", "low_collapse", "severe_collapse"]

    # Generate evaluation data
    x = torch.randint(0, 59, (100, 2))

    results = []

    for cond in conditions:
        cond_dir = os.path.join(base_dir, cond)
        if not os.path.exists(cond_dir):
            continue

        checkpoints = sorted([f for f in os.listdir(cond_dir) if f.startswith("checkpoint_")])

        # We'll plot heatmaps for specific milestones
        milestones = {}
        if cond == "pure":
            milestones = {500: "Pre-Grokking", 1500: "Grokking Onset", 5000: "Post-Grokking"}
        elif cond == "low_collapse":
            milestones = {1000: "Pre-Grokking", 3500: "Grokking Onset", 10000: "Post-Grokking"}
        elif cond == "severe_collapse":
            milestones = {1000: "Early", 5000: "Mid", 15000: "Late"}

        for ckpt in checkpoints:
            step = int(ckpt.split("_")[1].split(".")[0])
            ckpt_path = os.path.join(cond_dir, ckpt)

            data = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Init model
            model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512)
            # Remove module. prefix if present
            state_dict = data['model_state']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                attn_weights = get_attention_weights(model, x)
                entropy = compute_attention_entropy(attn_weights)
                results.append({"condition": cond, "step": step, "entropy": entropy})

                # Plot heatmap if it's a milestone
                # Find nearest milestone
                nearest_milestone = None
                for m_step, label in milestones.items():
                    if abs(step - m_step) <= 500: # Tolerance
                        nearest_milestone = label
                        break

                if nearest_milestone:
                    # Average over batch
                    avg_attn = attn_weights.mean(dim=0) # (heads, seq, seq)
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    for h in range(4):
                        sns.heatmap(avg_attn[h].numpy(), ax=axes[h], annot=True, cmap="YlGnBu", vmin=0, vmax=1)
                        axes[h].set_title(f"Head {h+1}")
                    plt.suptitle(f"Attention Patterns - {cond} ({nearest_milestone}, step {step})")
                    plt.savefig(os.path.join(out_dir, f"heatmap_{cond}_{nearest_milestone.replace(' ', '_').lower()}.png"))
                    plt.close()
                    # Remove from dict so we only plot once
                    keys_to_delete = [k for k, v in milestones.items() if v == nearest_milestone]
                    for k in keys_to_delete:
                        del milestones[k]

    # Plot entropy evolution
    import pandas as pd
    df = pd.DataFrame(results)
    if not df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="step", y="entropy", hue="condition")
        plt.title("Attention Entropy Evolution")
        plt.ylabel("Average Entropy (nats)")
        plt.xlabel("Step")
        plt.savefig(os.path.join(out_dir, "attention_entropy.png"))
        plt.savefig(os.path.join(out_dir, "attention_entropy.pdf"))
        plt.close()
        print("Attention visualizations generated.")

if __name__ == "__main__":
    analyze_checkpoints("results", "analysis/attention")
