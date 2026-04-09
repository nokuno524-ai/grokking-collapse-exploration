"""
Mechanistic analysis of grokking-collapse interplay.
Uses SAE-inspired feature analysis and data attribution to understand WHY collapse kills grokking.

Three-pronged approach:
1. Representation Analysis — train SAE on transformer hidden states, track feature evolution
2. Data Attribution — influence functions / TracIn to find which examples drive grokking
3. Circuit Analysis — track the Fourier circuit formation across conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from model import ModularArithmeticTransformer
from data import generate_modular_arithmetic, DatasetConfig


# ============================================================
# Part 1: Sparse Autoencoder for Feature Analysis
# ============================================================

class FeatureSAE(nn.Module):
    """Simple TopK SAE to decompose transformer hidden states."""
    def __init__(self, d_model: int, n_features: int, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        # Initialize decoder columns to unit norm
        nn.init.xavier_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_acts = self.encoder(x)
        # TopK activation
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        recon = self.decoder(acts)
        return recon, acts

    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon, acts = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        l0 = (acts > 0).float().mean()
        return {"loss": recon_loss, "recon_loss": recon_loss, "l0": l0}


def train_sae_on_hidden_states(
    model: ModularArithmeticTransformer,
    train_inputs: torch.Tensor,
    d_model: int = 128,
    n_features: int = 2048,
    k: int = 32,
    n_steps: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
) -> FeatureSAE:
    """Train an SAE on the transformer's hidden representations."""
    model = model.to(device).eval()
    sae = FeatureSAE(d_model, n_features, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Extract hidden states
    hidden_states = []
    with torch.no_grad():
        for i in range(0, len(train_inputs), batch_size):
            batch = train_inputs[i:i+batch_size].to(device)
            tok = model.token_embed(batch)
            positions = torch.arange(2, device=device).unsqueeze(0).expand(batch.shape[0], -1)
            pos = model.pos_embed(positions)
            h = tok + pos
            h = model.transformer(h)
            h = model.ln(h)
            hidden_states.append(h.cpu())
    hidden_states = torch.cat(hidden_states, dim=0)  # (N, 2, d_model)
    # Reshape to (N*2, d_model) — treat each position independently
    hidden_states = hidden_states.reshape(-1, d_model)
    dataset = TensorDataset(hidden_states)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for step in range(n_steps):
        for (batch,) in loader:
            batch = batch.to(device)
            losses = sae.loss(batch)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
        if step % 1000 == 0:
            print(f"  SAE step {step}: recon_loss={losses['recon_loss']:.6f}, l0={losses['l0']:.1f}")

    return sae


# ============================================================
# Part 2: Data Attribution via TracIn
# ============================================================

def compute_tracin_scores(
    model: ModularArithmeticTransformer,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    checkpoints: List[dict],
    lr: float = 1e-3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute TracIn-style influence scores: how much does each training example
    contribute to the loss on test examples?

    TracIn = sum over checkpoints of: grad_z_train · grad_z_test

    Returns: (n_test, n_train) influence matrix
    """
    model = model.to(device)
    n_train = len(train_inputs)
    n_test = len(test_inputs)

    influence = torch.zeros(n_test, n_train)

    for ckpt in checkpoints:
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Compute test gradients
        test_grads = []
        for i in range(n_test):
            x = test_inputs[i:i+1].to(device)
            y = test_targets[i:i+1].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
            grad_vec = torch.cat([g.flatten() for g in grad])
            test_grads.append(grad_vec)

        # Compute train gradients and dot product
        for j in range(n_train):
            x = train_inputs[j:j+1].to(device)
            y = train_targets[j:j+1].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
            train_grad_vec = torch.cat([g.flatten() for g in grad])

            for i in range(n_test):
                influence[i, j] += lr * (test_grads[i] @ train_grad_vec).item()

        print(f"  Checkpoint step {ckpt['step']}: influence computed")

    return influence


# ============================================================
# Part 3: Fourier Circuit Analysis
# ============================================================

def analyze_fourier_circuit(model: ModularArithmeticTransformer, prime: int = 59) -> Dict:
    """
    Analyze the Fourier structure of the embedding and attention layers.
    Based on Chan et al. (2023) — grokking involves learning discrete Fourier transform.
    """
    # Token embedding Fourier spectrum
    W = model.token_embed.weight.detach()  # (prime, d_model)
    spectrum = torch.fft.fft(W, dim=0).abs()  # (prime, d_model)

    # Average spectrum across embedding dims
    avg_spectrum = spectrum.mean(dim=1)  # (prime,)
    dc_component = avg_spectrum[0].item()
    freq_energy = avg_spectrum[1:]  # Exclude DC

    # Concentration metrics
    total_energy = freq_energy.sum().item()
    top1 = freq_energy.max().item() / total_energy
    top5 = freq_energy.topk(5).values.sum().item() / total_energy
    top10 = freq_energy.topk(10).values.sum().item() / total_energy

    # Dominant frequencies
    dominant_freqs = freq_energy.topk(5).indices.tolist()

    # Output head analysis
    W_out = model.output_head.weight.detach()  # (prime, d_model)
    out_spectrum = torch.fft.fft(W_out, dim=0).abs()
    out_avg = out_spectrum.mean(dim=1)
    out_freq_energy = out_avg[1:]
    out_total = out_freq_energy.sum().item()
    out_top5 = out_freq_energy.topk(5).values.sum().item() / out_total

    # Effective rank of embeddings
    svd_vals = torch.linalg.svdvals(W)
    svd_normed = svd_vals / svd_vals.sum()
    entropy = -(svd_normed * torch.log(svd_normed + 1e-10)).sum()
    eff_rank = torch.exp(entropy).item()

    return {
        "embedding_fourier_top1": top1,
        "embedding_fourier_top5": top5,
        "embedding_fourier_top10": top10,
        "dominant_frequencies": dominant_freqs,
        "output_fourier_top5": out_top5,
        "effective_rank": eff_rank,
        "dc_component": dc_component,
        "total_spectral_energy": total_energy,
    }


# ============================================================
# Part 4: Compare Conditions Across Training
# ============================================================

def compare_conditions_across_checkpoints(
    conditions: List[str],
    results_dir: str = "results",
    prime: int = 59,
    device: str = "cpu",
):
    """Load checkpoints from each condition and analyze circuit formation."""
    results = {}

    for cond in conditions:
        cond_dir = Path(results_dir) / cond
        if not cond_dir.exists():
            print(f"Skipping {cond}: no results dir")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing condition: {cond}")
        print(f"{'='*60}")

        # Load final results
        results_json = cond_dir / "results.json"
        if results_json.exists():
            with open(results_json) as f:
                cond_results = json.load(f)

        # Analyze each checkpoint
        checkpoint_analysis = []
        ckpt_files = sorted(cond_dir.glob("checkpoint_*.pt"))
        for ckpt_file in ckpt_files:
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            step = ckpt["step"]

            model = ModularArithmeticTransformer(prime=prime)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            fourier = analyze_fourier_circuit(model, prime)

            # Also check weight norms per layer
            embed_norm = model.token_embed.weight.norm().item()
            output_norm = model.output_head.weight.norm().item()
            transformer_params = list(model.transformer.parameters())
            transformer_norm = sum(p.norm().item()**2 for p in transformer_params)**0.5

            analysis = {
                "step": step,
                "fourier": fourier,
                "embed_norm": embed_norm,
                "output_norm": output_norm,
                "transformer_norm": transformer_norm,
            }
            checkpoint_analysis.append(analysis)

            print(f"  Step {step:6d}: fourier_top5={fourier['embedding_fourier_top5']:.3f} "
                  f"rank={fourier['effective_rank']:.1f} "
                  f"dom_freqs={fourier['dominant_frequencies']}")

        results[cond] = {
            "checkpoint_analysis": checkpoint_analysis,
            "final_results": cond_results if results_json.exists() else None,
        }

    return results


# ============================================================
# Part 5: Full Pipeline
# ============================================================

def run_full_analysis(
    conditions: List[str] = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"],
    results_dir: str = "results",
    output_dir: str = "analysis_output",
    prime: int = 59,
    device: str = "cpu",
):
    """Run the full mechanistic analysis pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Compare Fourier circuit formation across conditions
    print("\n" + "="*60)
    print("STEP 1: Fourier Circuit Analysis Across Conditions")
    print("="*60)
    comparison = compare_conditions_across_checkpoints(conditions, results_dir, prime, device)

    # Step 2: Train SAEs on pure vs severe_collapse final checkpoints
    print("\n" + "="*60)
    print("STEP 2: SAE Feature Analysis (Pure vs Severe Collapse)")
    print("="*60)

    sae_results = {}
    for cond in ["pure", "severe_collapse"]:
        ckpt_file = Path(results_dir) / cond / "checkpoint_50000.pt"
        if not ckpt_file.exists():
            continue
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        model = ModularArithmeticTransformer(prime=prime)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Generate data for this condition
        config = DatasetConfig(prime=prime, seed=42)
        train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

        print(f"\nTraining SAE on {cond} hidden states...")
        sae = train_sae_on_hidden_states(
            model, train_in, d_model=128, n_features=2048, k=32,
            n_steps=3000, device=device,
        )

        # Analyze feature activations across all data
        with torch.no_grad():
            tok = model.token_embed(train_in.to(device))
            positions = torch.arange(2, device=device).unsqueeze(0).expand(train_in.shape[0], -1)
            pos = model.pos_embed(positions)
            h = tok + pos
            h = model.transformer(h)
            h = model.ln(h)
            h_flat = h.reshape(-1, 128)
            _, acts = sae(h_flat)

        # Feature statistics
        feature_freq = (acts > 0).float().mean(dim=0)
        dead_features = (feature_freq < 1e-4).sum().item()
        active_features = (feature_freq > 0.01).sum().item()

        sae_results[cond] = {
            "dead_features": dead_features,
            "active_features": active_features,
            "total_features": 2048,
            "mean_activation_freq": feature_freq.mean().item(),
            "max_activation_freq": feature_freq.max().item(),
            "feature_freq_distribution": {
                "p25": torch.quantile(feature_freq[feature_freq > 0], 0.25).item() if (feature_freq > 0).any() else 0,
                "p50": torch.quantile(feature_freq[feature_freq > 0], 0.50).item() if (feature_freq > 0).any() else 0,
                "p75": torch.quantile(feature_freq[feature_freq > 0], 0.75).item() if (feature_freq > 0).any() else 0,
            }
        }
        print(f"  {cond}: {dead_features} dead, {active_features} active features")

    # Step 3: Data attribution (TracIn) — compare pure vs collapsed
    print("\n" + "="*60)
    print("STEP 3: Data Attribution (TracIn)")
    print("="*60)

    attribution_results = {}
    for cond in ["pure", "medium_collapse"]:
        ckpt_files = sorted((Path(results_dir) / cond).glob("checkpoint_*.pt"))
        if len(ckpt_files) < 3:
            continue

        # Load a few checkpoints for TracIn
        checkpoints = []
        for f in ckpt_files[::3]:  # every 3rd checkpoint
            ckpt = torch.load(f, map_location=device, weights_only=False)
            checkpoints.append(ckpt)

        # Generate data
        collapse_levels = {
            "pure": 0.0,
            "medium_collapse": 0.15,
        }
        config = DatasetConfig(prime=prime, collapse_level=collapse_levels.get(cond, 0.0),
                               collapse_severity=0.5, seed=42)
        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

        # Sample for tractability
        n_test_sample = min(100, len(test_in))
        n_train_sample = min(200, len(train_in))

        print(f"Computing TracIn for {cond} ({len(checkpoints)} checkpoints, "
              f"{n_test_sample} test × {n_train_sample} train)...")

        influence = compute_tracin_scores(
            ModularArithmeticTransformer(prime=prime),
            train_in[:n_train_sample],
            train_tgt[:n_train_sample],
            test_in[:n_test_sample],
            test_tgt[:n_test_sample],
            checkpoints,
            device=device,
        )

        # Identify top influential examples
        mean_influence = influence.mean(dim=0)  # (n_train,)
        top_influential = mean_influence.topk(10)

        attribution_results[cond] = {
            "mean_influence_std": mean_influence.std().item(),
            "max_influence": mean_influence.max().item(),
            "min_influence": mean_influence.min().item(),
            "top_influential_indices": top_influential.indices.tolist(),
            "top_influential_scores": top_influential.values.tolist(),
        }
        print(f"  {cond}: max influence={mean_influence.max():.4f}, "
              f"std={mean_influence.std():.4f}")

    # Save all results
    full_output = {
        "fourier_comparison": {
            cond: {
                "checkpoints": [
                    {"step": a["step"], "fourier": a["fourier"],
                     "embed_norm": a["embed_norm"], "output_norm": a["output_norm"]}
                    for a in data["checkpoint_analysis"]
                ],
                "final": data["final_results"],
            }
            for cond, data in comparison.items()
        },
        "sae_analysis": sae_results,
        "attribution": attribution_results,
    }

    output_file = output_path / "mechanistic_analysis.json"
    with open(output_file, "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    print(f"\nFull analysis saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Why Collapse Kills Grokking")
    print("="*60)
    for cond, data in comparison.items():
        if data["checkpoint_analysis"]:
            first = data["checkpoint_analysis"][0]
            last = data["checkpoint_analysis"][-1]
            print(f"\n{cond}:")
            print(f"  Fourier top5: {first['fourier']['embedding_fourier_top5']:.3f} → {last['fourier']['embedding_fourier_top5']:.3f}")
            print(f"  Eff rank: {first['fourier']['effective_rank']:.1f} → {last['fourier']['effective_rank']:.1f}")
            print(f"  Dominant freqs: {last['fourier']['dominant_frequencies']}")

    return full_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prime", type=int, default=59)
    args = parser.parse_args()

    run_full_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        device=args.device,
        prime=args.prime,
    )
