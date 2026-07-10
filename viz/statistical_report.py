import json
import numpy as np
import torch
import scipy.stats as stats
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

def compute_kl_divergence(p, q):
    """Compute KL divergence KL(P || Q). Assumes p, q are probability distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Add small epsilon to avoid div by zero or log zero
    p = p + 1e-10
    q = q + 1e-10

    p = p / p.sum()
    q = q / q.sum()

    return np.sum(p * np.log(p / q))

def get_layer_weights(model):
    """Extract weight distributions from key layers."""
    weights = {
        'embedding': model.token_embed.weight.detach().cpu().numpy().flatten(),
        'attn_out': model.transformer.layers[0].self_attn.out_proj.weight.detach().cpu().numpy().flatten(),
        'ffn_out': model.transformer.layers[0].linear2.weight.detach().cpu().numpy().flatten()
    }
    return weights

def generate_statistical_report(results_dir: str = "results", output_dir: str = "viz"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Compare pure vs severe collapse final weights
    pure_dir = results_path / "pure"
    severe_dir = results_path / "severe_collapse"

    pure_ckpts = sorted(list(pure_dir.glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))
    severe_ckpts = sorted(list(severe_dir.glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))

    if not pure_ckpts or not severe_ckpts:
        print("Missing checkpoints, cannot generate statistical report")
        return

    pure_model = load_model(pure_ckpts[-1])
    severe_model = load_model(severe_ckpts[-1])

    pure_weights = get_layer_weights(pure_model)
    severe_weights = get_layer_weights(severe_model)

    # 1. Weight Distributions Histogram Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    layers = ['embedding', 'attn_out', 'ffn_out']
    titles = ['Token Embedding Weights', 'Attention Out Projection', 'FFN Out Projection']

    report_lines = ["# Statistical Analysis Report\n"]
    report_lines.append("## Kolmogorov-Smirnov Test for Weight Distributions (Pure vs Severe Collapse)")
    report_lines.append("| Layer | KS Statistic | p-value |")
    report_lines.append("|---|---|---|")

    for i, (layer, title) in enumerate(zip(layers, titles)):
        pw = pure_weights[layer]
        sw = severe_weights[layer]

        axs[i].hist(pw, bins=50, alpha=0.5, label='Pure Data', density=True, color='#2ecc71')
        axs[i].hist(sw, bins=50, alpha=0.5, label='Severe Collapse', density=True, color='#8e44ad')
        axs[i].set_title(title)
        axs[i].legend()

        # KS Test
        ks_stat, p_val = stats.ks_2samp(pw, sw)
        report_lines.append(f"| {title} | {ks_stat:.4f} | {p_val:.4e} |")

    plt.tight_layout()
    plt.savefig(out_path / "weight_distributions.png", dpi=300)
    print("Saved weight distributions plot to viz/weight_distributions.png")

    # 2. Fourier KL Divergence Analysis
    report_lines.append("\n## KL Divergence of Fourier Spectrums")
    report_lines.append("Measuring how much the learned algorithm's spectrum diverges from Pure Data.")
    report_lines.append("| Condition | KL(Condition || Pure) | KL(Pure || Condition) |")
    report_lines.append("|---|---|---|")

    # Get pure spectrum
    pure_W = pure_model.token_embed.weight.detach()
    pure_spec = torch.fft.fft(pure_W, dim=0).abs().pow(2).mean(dim=1).numpy()
    pure_spec = pure_spec[:pure_W.shape[0]//2 + 1]

    for cond in SEVERITY_ORDER:
        cond_dir = results_path / cond
        if not cond_dir.exists():
            continue

        ckpts = sorted(list(cond_dir.glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            model = load_model(ckpts[-1])
            W = model.token_embed.weight.detach()
            spec = torch.fft.fft(W, dim=0).abs().pow(2).mean(dim=1).numpy()
            spec = spec[:W.shape[0]//2 + 1]

            kl_c_p = compute_kl_divergence(spec, pure_spec)
            kl_p_c = compute_kl_divergence(pure_spec, spec)

            report_lines.append(f"| {cond} | {kl_c_p:.4f} | {kl_p_c:.4f} |")

    # Gradient Flow Analysis
    report_lines = add_gradient_flow_to_report(results_path, out_path, report_lines)

    # Gradient Flow Analysis
    report_lines = add_gradient_flow_to_report(results_path, out_path, report_lines)

    # Save Report
    with open(out_path / "statistical_report.md", "w") as f:
        f.write("\n".join(report_lines))
    print("Saved statistical report to viz/statistical_report.md")

if __name__ == "__main__":
    generate_statistical_report()

def analyze_gradient_flow(model):
    """Placeholder for gradient flow analysis on the loaded model."""
    # Since we load from static checkpoints without gradients,
    # we can compute a proxy of gradient flow by analyzing weight variances/norms
    # or consecutive weight changes if multiple checkpoints were passed.
    # Here, we will compute the relative weight norm per layer as a proxy for structural learning.
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norms[name] = param.norm().item()
    return norms

def add_gradient_flow_to_report(results_path, out_path, report_lines):
    report_lines.append("\n## Gradient Flow Proxy (Final Weight Norms per Layer)")
    report_lines.append("| Layer | Pure Data Norm | Severe Collapse Norm |")
    report_lines.append("|---|---|---|")

    pure_ckpts = sorted(list((results_path / "pure").glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))
    severe_ckpts = sorted(list((results_path / "severe_collapse").glob("checkpoint_*.pt")), key=lambda p: int(p.stem.split("_")[1]))

    if not pure_ckpts or not severe_ckpts:
        return report_lines

    pure_model = load_model(pure_ckpts[-1])
    severe_model = load_model(severe_ckpts[-1])

    p_norms = analyze_gradient_flow(pure_model)
    s_norms = analyze_gradient_flow(severe_model)

    for layer_name in p_norms:
        report_lines.append(f"| {layer_name} | {p_norms[layer_name]:.4f} | {s_norms[layer_name]:.4f} |")

    return report_lines

# Update generate_statistical_report to include gradient flow
with open(__file__, 'r') as f:
    content = f.read()

import re
content = re.sub(
    r'    # Save Report\n    with open\(out_path / "statistical_report\.md", "w"\) as f:\n        f\.write\("\\n"\.join\(report_lines\)\)',
    r'    # Gradient Flow Analysis\n    report_lines = add_gradient_flow_to_report(results_path, out_path, report_lines)\n\n    # Save Report\n    with open(out_path / "statistical_report.md", "w") as f:\n        f.write("\\n".join(report_lines))',
    content
)

with open(__file__, 'w') as f:
    f.write(content)
