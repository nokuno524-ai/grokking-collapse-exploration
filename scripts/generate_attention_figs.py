import torch
from pathlib import Path
from src.analysis.attention_viz import plot_attention_entropy_trajectory, plot_head_specialization_trajectory, plot_attention_heatmap

output_dir = Path("analysis/attention")
data_path = output_dir / "extracted_attention.pt"

if not data_path.exists():
    print(f"File {data_path} not found.")
    exit(1)

attention_data = torch.load(data_path, weights_only=False)

# 1. Entropy trajectory
fig = plot_attention_entropy_trajectory(attention_data, layer_idx=0)
fig.savefig(output_dir / "attention_entropy_trajectory.png")
print("Saved entropy trajectory")

# 2. Head specialization trajectory
fig = plot_head_specialization_trajectory(attention_data, layer_idx=0)
fig.savefig(output_dir / "head_specialization_trajectory.png")
print("Saved head specialization trajectory")

# 3. Heatmaps for pure vs severe collapse at step 50000 (if exists)
for cond in ["pure", "severe_collapse"]:
    if cond in attention_data and 50000 in attention_data[cond]:
        attn_weights = attention_data[cond][50000]
        fig = plot_attention_heatmap(attn_weights, layer_idx=0, batch_idx=0, head_idx=0)
        fig.savefig(output_dir / f"attention_heatmap_{cond}_step50000.png")
        print(f"Saved heatmap for {cond}")
