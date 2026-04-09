"""
Deep data attribution investigation.
Focus: which specific training examples are poison pills for grokking?
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from data import generate_modular_arithmetic, DatasetConfig, apply_collapse
from model import ModularArithmeticTransformer
from collections import Counter


def identify_collapsed_examples(prime=59, seed=42):
    """Find exactly which training examples had their labels corrupted."""
    rng = np.random.RandomState(seed)

    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    all_targets = [(a + b) % prime for a, b in all_pairs]

    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * 0.3)
    train_idx = indices[:n_train]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets = [all_targets[i] for i in train_idx]

    results = {}
    for name, collapse_level, severity in [
        ("medium", 0.15, 0.5),
        ("high", 0.30, 0.7),
        ("severe", 0.50, 0.9),
    ]:
        n_replace = int(len(train_targets) * collapse_level)
        replace_idx = set(rng.choice(len(train_targets), n_replace, replace=False).tolist())

        # Apply collapse to get the actual targets
        collapsed = list(train_targets)
        # Reproduce the collapse logic
        target_counts = Counter(train_targets)
        total = len(train_targets)
        freq = {t: c / total for t, c in target_counts.items()}
        temp = max(0.1, 1.0 - severity)
        collapsed_probs = {t: freq.get(t, 1.0/prime) ** (1.0/temp) for t in range(prime)}
        total_prob = sum(collapsed_probs.values())
        collapsed_probs = {t: p/total_prob for t, p in collapsed_probs.items()}
        ct = list(collapsed_probs.keys())
        cw = [collapsed_probs[t] for t in ct]

        corrupted = {}
        for idx in replace_idx:
            new_target = int(rng.choice(ct, p=cw))
            original = train_targets[idx]
            if new_target != original:
                corrupted[idx] = {
                    "pair": train_pairs[idx],
                    "original_target": original,
                    "corrupted_target": new_target,
                    "error": (new_target - original) % prime,
                }

        results[name] = {
            "total_train": len(train_targets),
            "n_corrupted": len(corrupted),
            "collapse_level": collapse_level,
            "corrupted_indices": list(corrupted.keys()),
            "corruption_details": corrupted,
            "target_distribution_shift": {
                str(t): collapsed_probs[t] / (1.0/prime)  # fold change vs uniform
                for t in range(prime)
            },
        }
        print(f"\n{name} collapse ({collapse_level*100:.0f}%):")
        print(f"  {len(corrupted)} actually corrupted examples (label changed)")
        print(f"  Error magnitudes: mean={np.mean([abs(v['error']) for v in corrupted.values()]):.1f}, "
              f"max={max(abs(v['error']) for v in corrupted.values())}")

    return results


def grad_based_attribution(
    condition_name: str,
    collapse_level: float,
    collapse_severity: float,
    checkpoint_path: str,
    prime: int = 59,
    device: str = "cpu",
):
    """
    For each training example, compute:
    1. Loss gradient norm (how much it pushes the model)
    2. Gradient direction similarity to the "correct" gradient
    3. Whether the example is correct or corrupted
    """
    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ModularArithmeticTransformer(prime=prime).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Generate data
    config = DatasetConfig(prime=prime, collapse_level=collapse_level,
                           collapse_severity=collapse_severity, seed=42)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # Also generate CLEAN targets for comparison
    clean_config = DatasetConfig(prime=prime, collapse_level=0.0, seed=42)
    _, clean_tgt, _, _ = generate_modular_arithmetic(clean_config)

    # Identify which examples are corrupted
    is_corrupted = (train_tgt != clean_tgt[:len(train_tgt)])

    # Compute per-example gradient norms and loss
    grad_norms = []
    losses = []
    model.eval()

    # Compute "ideal" gradient (gradient on clean test loss)
    model.zero_grad()
    test_logits = model(test_in[:200].to(device))
    test_loss = F.cross_entropy(test_logits, test_tgt[:200].to(device))
    test_grad = torch.autograd.grad(test_loss, model.parameters(), retain_graph=True)
    test_grad_vec = torch.cat([g.flatten() for g in test_grad])
    test_grad_vec = test_grad_vec / (test_grad_vec.norm() + 1e-8)

    for i in range(len(train_in)):
        model.zero_grad()
        x = train_in[i:i+1].to(device)
        y = train_tgt[i:i+1].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, model.parameters())
        grad_vec = torch.cat([g.flatten() for g in grad])
        grad_norm = grad_vec.norm().item()
        cosine_sim = F.cosine_similarity(grad_vec.unsqueeze(0), test_grad_vec.unsqueeze(0)).item()

        grad_norms.append(grad_norm)
        losses.append(loss.item())

    grad_norms = np.array(grad_norms)
    losses = np.array(losses)
    is_corrupted = is_corrupted.numpy()

    analysis = {
        "condition": condition_name,
        "collapse_level": collapse_level,
        "n_corrupted": int(is_corrupted.sum()),
        "n_correct": int((~is_corrupted).sum()),
        "corrupted_grad_norm_mean": float(grad_norms[is_corrupted].mean()) if is_corrupted.any() else 0,
        "correct_grad_norm_mean": float(grad_norms[~is_corrupted].mean()) if (~is_corrupted).any() else 0,
        "corrupted_loss_mean": float(losses[is_corrupted].mean()) if is_corrupted.any() else 0,
        "correct_loss_mean": float(losses[~is_corrupted].mean()) if (~is_corrupted).any() else 0,
        "top_grad_norm_indices": np.argsort(grad_norms)[-20:].tolist(),
        "top_loss_indices": np.argsort(losses)[-20:].tolist(),
    }

    print(f"\n{condition_name} ({collapse_level*100:.0f}% collapse):")
    print(f"  Corrupted examples: {analysis['n_corrupted']}")
    print(f"  Corrupted grad norm: {analysis['corrupted_grad_norm_mean']:.4f}")
    print(f"  Correct grad norm:   {analysis['correct_grad_norm_mean']:.4f}")
    print(f"  Corrupted loss: {analysis['corrupted_loss_mean']:.4f}")
    print(f"  Correct loss:   {analysis['correct_loss_mean']:.4f}")

    return analysis


def run_data_attribution_study(
    results_dir: str = "results",
    output_dir: str = "analysis_output",
    prime: int = 59,
    device: str = "cpu",
):
    """Full data attribution pipeline."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Identify corrupted examples
    print("="*60)
    print("STEP 1: Identifying corrupted training examples")
    print("="*60)
    corruption_map = identify_collapsed_examples(prime)

    # Step 2: Gradient-based attribution at final checkpoint
    print("\n" + "="*60)
    print("STEP 2: Gradient attribution at final checkpoint")
    print("="*60)

    attribution = {}
    for cond, collapse_level, severity in [
        ("pure", 0.0, 0.0),
        ("low_collapse", 0.05, 0.3),
        ("medium_collapse", 0.15, 0.5),
        ("high_collapse", 0.30, 0.7),
        ("severe_collapse", 0.50, 0.9),
    ]:
        ckpt_path = Path(results_dir) / cond / "checkpoint_50000.pt"
        if not ckpt_path.exists():
            continue
        attribution[cond] = grad_based_attribution(
            cond, collapse_level, severity, str(ckpt_path), prime, device
        )

    # Step 3: Track attribution over time (key checkpoints)
    print("\n" + "="*60)
    print("STEP 3: Attribution evolution across training")
    print("="*60)

    time_series = {}
    for cond, collapse_level, severity in [("medium_collapse", 0.15, 0.5)]:
        cond_dir = Path(results_dir) / cond
        ckpts = sorted(cond_dir.glob("checkpoint_*.pt"))

        for ckpt_file in ckpts:
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            step = ckpt["step"]
            model = ModularArithmeticTransformer(prime=prime).to(device)
            model.load_state_dict(ckpt["model_state"])

            config = DatasetConfig(prime=prime, collapse_level=collapse_level,
                                   collapse_severity=severity, seed=42)
            train_in, train_tgt, _, _ = generate_modular_arithmetic(config)
            clean_config = DatasetConfig(prime=prime, collapse_level=0.0, seed=42)
            _, clean_tgt, _, _ = generate_modular_arithmetic(clean_config)
            is_corrupted = (train_tgt != clean_tgt[:len(train_tgt)])

            # Compute losses for corrupted vs clean
            model.eval()
            with torch.no_grad():
                logits = model(train_in.to(device))
                all_losses = F.cross_entropy(logits, train_tgt.to(device), reduction='none').cpu()

            corrupted_loss = all_losses[is_corrupted].mean().item() if is_corrupted.any() else 0
            correct_loss = all_losses[~is_corrupted].mean().item() if (~is_corrupted).any() else 0

            # Accuracy on corrupted vs clean targets
            preds = logits.argmax(dim=-1).cpu()
            acc_on_corrupted = (preds == train_tgt.to(device)).cpu()
            acc_corrupted = acc_on_corrupted[is_corrupted].float().mean().item() if is_corrupted.any() else 0
            acc_correct = acc_on_corrupted[~is_corrupted].float().mean().item() if (~is_corrupted).any() else 0

            time_series[f"step_{step}"] = {
                "step": step,
                "corrupted_loss": corrupted_loss,
                "correct_loss": correct_loss,
                "loss_gap": corrupted_loss - correct_loss,
                "acc_on_corrupted_labels": acc_corrupted,
                "acc_on_correct_labels": acc_correct,
            }
            print(f"  Step {step:6d}: corrupted_loss={corrupted_loss:.4f} correct_loss={correct_loss:.4f} "
                  f"gap={corrupted_loss-correct_loss:+.4f} acc_corrupted={acc_corrupted:.3f} acc_correct={acc_correct:.3f}")

    # Save
    output = {
        "corruption_map": corruption_map,
        "final_attribution": attribution,
        "medium_collapse_time_series": time_series,
    }

    out_file = Path(output_dir) / "data_attribution.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {out_file}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_data_attribution_study(args.results_dir, args.output_dir, device=args.device)
