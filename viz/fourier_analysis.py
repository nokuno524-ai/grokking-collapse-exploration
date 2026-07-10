import json
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import ModularArithmeticTransformer

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

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

def get_fourier_spectrum(model):
    """Get the squared magnitude of the DFT of embedding matrix as required."""
    W = model.token_embed.weight.detach() # (prime, d_model)
    # The memory instruction: "When computing the Fourier spectrum of embeddings in this repo, use `.abs() ** 2` to obtain the squared magnitude (energy) instead of just the absolute magnitude."
    spectrum = torch.fft.fft(W, dim=0).abs() ** 2

    # We only care about frequencies up to prime // 2 due to symmetry
    prime = W.shape[0]
    half_prime = prime // 2 + 1

    # Average across embedding dimension to get power per frequency
    power = spectrum[:half_prime].mean(dim=1).numpy()

    return power

def create_fourier_visualization(results_dir: str = "results", output_dir: str = "viz"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    # Panel 1: Final Fourier Spectrum (Power across frequencies)
    ax = axs[0]
    for cond in SEVERITY_ORDER:
        cond_dir = results_path / cond
        if not cond_dir.exists():
            continue

        ckpts = sorted(list(cond_dir.glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            final_ckpt = ckpts[-1]
            try:
                model = load_model(final_ckpt)
                power = get_fourier_spectrum(model)
                freqs = np.arange(len(power))

                # Normalize power for easier comparison
                norm_power = power / power.sum()
                ax.plot(freqs, norm_power, label=cond.replace("_", " ").title(), color=colors.get(cond, "gray"), linewidth=2)
            except Exception as e:
                print(f"Error processing {cond}: {e}")

    ax.set_title("Final Embedding Fourier Spectrum")
    ax.set_xlabel("Frequency (k)")
    ax.set_ylabel("Normalized Power")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Concentration Evolution over time
    ax = axs[1]
    for cond in SEVERITY_ORDER:
        cond_dir = results_path / cond
        json_path = cond_dir / "results.json"

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)

            if "history" in data:
                steps = [e["step"] for e in data["history"]]
                conc = [e.get("fourier_concentration", 0) for e in data["history"]]
                ax.plot(steps, conc, label=cond.replace("_", " ").title(), color=colors.get(cond, "gray"), linewidth=2)

    ax.set_title("Fourier Concentration over Training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fourier Concentration")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "fourier_analysis.png", dpi=300)
    print(f"Fourier visualization saved to {out_path}/fourier_analysis.png")

if __name__ == "__main__":
    create_fourier_visualization()
