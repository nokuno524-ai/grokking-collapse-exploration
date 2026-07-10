import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import ModularArithmeticTransformer

def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get("config", {})
    class DummyConfig:
        prime = config.get("prime", 59)
        d_model = config.get("d_model", 128)
        n_heads = config.get("n_heads", 4)
        d_ff = config.get("d_ff", 512)
        n_layers = config.get("n_layers", 1)

    cfg = DummyConfig()
    model = ModularArithmeticTransformer(
        prime=cfg.prime,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def extract_attention(model, x, y, p=59):
    with torch.no_grad():
        token_a = torch.tensor([[x]])
        token_b = torch.tensor([[y]])
        inputs = torch.cat([token_a, token_b], dim=1) # The model takes (batch, 2)

        tok = model.token_embed(inputs)
        positions = torch.arange(2, device=inputs.device).unsqueeze(0)
        pos = model.pos_embed(positions)
        h = tok + pos

        layer = model.transformer.layers[0]
        attn_output, attn_weight = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
        return attn_weight[0].detach().cpu().numpy() # (n_heads, 2, 2)

def create_animation(results_dir: str = "results", output_dir: str = "viz"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pure_dir = results_path / "pure"
    severe_dir = results_path / "severe_collapse"

    pure_ckpts = sorted(list(pure_dir.glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))
    severe_ckpts = sorted(list(severe_dir.glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))

    if not pure_ckpts or not severe_ckpts:
        print("Missing checkpoints for pure or severe_collapse")
        return

    steps = [int(p.stem.split("_")[1]) for p in pure_ckpts]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle("Attention Pattern Evolution (Head 0)")

    im_pure = axs[0].imshow(np.zeros((2, 2)), vmin=0, vmax=1, cmap="viridis")
    axs[0].set_title("Pure Data")
    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[0].set_xticklabels(["a", "b"])
    axs[0].set_yticklabels(["a", "b"])

    im_severe = axs[1].imshow(np.zeros((2, 2)), vmin=0, vmax=1, cmap="viridis")
    axs[1].set_title("Severe Collapse")
    axs[1].set_xticks([0, 1])
    axs[1].set_yticks([0, 1])
    axs[1].set_xticklabels(["a", "b"])
    axs[1].set_yticklabels(["a", "b"])

    fig.colorbar(im_pure, ax=axs.ravel().tolist(), orientation="vertical")

    def update(frame_idx):
        if frame_idx < len(pure_ckpts):
            p_ckpt = pure_ckpts[frame_idx]
            s_ckpt = severe_ckpts[min(frame_idx, len(severe_ckpts)-1)]
            step = steps[frame_idx]

            p_model = load_model(p_ckpt)
            s_model = load_model(s_ckpt)

            p_attn = extract_attention(p_model, 3, 5)
            s_attn = extract_attention(s_model, 3, 5)

            if len(p_attn.shape) == 3:
                im_pure.set_array(p_attn[0])
                im_severe.set_array(s_attn[0])
            else:
                im_pure.set_array(p_attn)
                im_severe.set_array(s_attn)

            fig.suptitle(f"Attention Pattern Evolution - Step {step}")

        return [im_pure, im_severe]

    ani = animation.FuncAnimation(fig, update, frames=len(pure_ckpts), blit=False, repeat=False)

    # Save as GIF
    gif_path = out_path / "attention_evolution.gif"
    try:
        writer = animation.PillowWriter(fps=2)
        ani.save(gif_path, writer=writer)
        print(f"Animation saved to {gif_path}")
    except Exception as e:
        print(f"Could not save GIF: {e}")

    # Save as MP4
    mp4_path = out_path / "attention_evolution.mp4"
    try:
        writer = animation.FFMpegWriter(fps=2)
        ani.save(mp4_path, writer=writer)
        print(f"Animation saved to {mp4_path}")
    except Exception as e:
        print(f"Could not save MP4: {e}. Skipping MP4 generation.")

if __name__ == "__main__":
    create_animation()
