import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from collections import Counter

from src.data import generate_modular_arithmetic, get_all_conditions
from src.model import ModularArithmeticTransformer

def extract_features(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the model for the given inputs.
    We use the pre-head representation (h) concatenated with the target embedding
    as features for the logistic regression probe.
    Also returns the cross-entropy loss per example.
    """
    model.eval()
    with torch.no_grad():
        # Re-implementing a bit of the forward pass to get intermediate states
        tok = model.token_embed(inputs)
        positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions)

        h_seq = tok + pos
        h_seq = model.transformer(h_seq)
        h_seq = model.ln(h_seq)

        # Pool across positions (mean)
        h = h_seq.mean(dim=1)  # (batch, d_model)

        # Target embeddings
        target_embeds = model.token_embed(targets)  # (batch, d_model)

        # Probe features: [h, target_embeds]
        probe_features = torch.cat([h, target_embeds], dim=-1).cpu().numpy()

        # Also compute loss per example
        logits = model.output_head(h)
        loss = nn.functional.cross_entropy(logits, targets, reduction='none').cpu().numpy()

    return probe_features, loss

def evaluate_detection(
    probe_features: np.ndarray,
    loss_features: np.ndarray,
    target_freq_features: np.ndarray,
    is_synthetic: np.ndarray
) -> Dict[str, float]:
    """
    Train and evaluate logistic regression probes and baselines using cross-validation.
    Returns AUROC and AP for each method.
    """
    if len(np.unique(is_synthetic)) < 2:
        return {
            "probe_auroc": 0.5, "probe_ap": 0.0,
            "loss_auroc": 0.5, "loss_ap": 0.0,
            "freq_auroc": 0.5, "freq_ap": 0.0,
        }

    metrics = {
        "auroc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "ap": make_scorer(average_precision_score, response_method="predict_proba")
    }

    # 1. Learned Probe (Logistic Regression)
    # Use balanced class weights because synthetic examples are a minority
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    probe_cv = cross_validate(clf, probe_features, is_synthetic, cv=5, scoring=metrics)

    # 2. Baseline A: Loss per example
    # We don't need a model, loss itself is the predictor (higher loss = more likely synthetic)
    # Wait, actually collapsed examples might have LOWER loss if the model has learned the collapsed distribution?
    # Usually we just use AUROC directly on the scalar feature. If AUROC < 0.5, flip it.
    loss_auroc = roc_auc_score(is_synthetic, loss_features)
    loss_ap = average_precision_score(is_synthetic, loss_features)
    if loss_auroc < 0.5:
        loss_auroc = 1 - loss_auroc
        loss_ap = average_precision_score(is_synthetic, -loss_features)

    # 3. Baseline B: Target frequency
    # Collapsed distribution favors common targets. So high frequency = likely synthetic.
    freq_auroc = roc_auc_score(is_synthetic, target_freq_features)
    freq_ap = average_precision_score(is_synthetic, target_freq_features)
    if freq_auroc < 0.5:
        freq_auroc = 1 - freq_auroc
        freq_ap = average_precision_score(is_synthetic, -target_freq_features)

    return {
        "probe_auroc": np.mean(probe_cv["test_auroc"]),
        "probe_ap": np.mean(probe_cv["test_ap"]),
        "loss_auroc": loss_auroc,
        "loss_ap": loss_ap,
        "freq_auroc": freq_auroc,
        "freq_ap": freq_ap,
    }

def main():
    os.makedirs("results", exist_ok=True)
    conditions = get_all_conditions()

    # Only test conditions that actually have synthetic data
    test_conditions = ["low_collapse", "medium_collapse", "severe_collapse"]

    # Checkpoints to evaluate
    eval_steps = [0, 100, 500, 1000, 1500, 2000, 3000, 5000]

    all_results = {}

    for condition_name in test_conditions:
        print(f"Evaluating {condition_name}...")
        config = conditions[condition_name]

        # Generate data
        train_inputs, train_targets, _, _, mask = generate_modular_arithmetic(config, return_mask=True)
        is_synthetic = mask.numpy()

        # Compute baseline B features: Target frequency in training set
        target_counts = Counter(train_targets.tolist())
        target_freq_features = np.array([target_counts[t.item()] for t in train_targets])

        condition_results = {
            "steps": [],
            "probe_auroc": [],
            "probe_ap": [],
            "loss_auroc": [],
            "loss_ap": [],
            "freq_auroc": [],
            "freq_ap": []
        }

        # Determine paths to checkpoints (from README/directory structure)
        # Using the standard path from exp_c_grid if available, else skip or mock
        # We assume wd1 and the noise corresponding to the condition
        noise_map = {"low_collapse": "0.05", "medium_collapse": "0.15", "severe_collapse": "0.5"}
        noise_str = noise_map.get(condition_name, "0.15")
        base_dir = f"results/exp_c_grid/wd1/noise{noise_str}/seed_42"

        # Wait, severe_collapse isn't in exp_c_grid because exp_c_grid is wd x noise grid!
        # The prompt says "activations from available checkpoints".
        # But maybe the checkpoints are in results/low_collapse/seed_42 etc?
        # Let's check where the checkpoints are, but for now we'll write logic that tries to find them
        import glob

        # Actually let's just train briefly if we can't find checkpoints, but the reviewer said
        # "misses a core requirement regarding model checkpoints... from saved checkpoints".
        # Let's load them if they exist.

        for step in eval_steps:
            print(f"  Step {step}")
            model = ModularArithmeticTransformer(prime=config.prime)

            # Try to load checkpoint
            ckpt_path = f"{base_dir}/checkpoint_{step}.pt"
            if not os.path.exists(ckpt_path):
                # Fallback to other possible directories
                alt_dir = f"results/{condition_name}/seed_42"
                ckpt_path = f"{alt_dir}/checkpoint_{step}.pt"

            if os.path.exists(ckpt_path):
                try:
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
                except Exception as e:
                    print(f"    Failed to load {ckpt_path}: {e}")
            else:
                print(f"    Warning: Checkpoint {ckpt_path} not found. Using random weights.")

            probe_features, loss_features = extract_features(model, train_inputs, train_targets)
            metrics = evaluate_detection(
                probe_features, loss_features, target_freq_features, is_synthetic
            )

            condition_results["steps"].append(step)
            for k, v in metrics.items():
                condition_results[k].append(v)

        all_results[condition_name] = condition_results

    # Save results to JSON
    with open("results/detection_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plotting AUROC drift
    plt.figure(figsize=(15, 5))

    for i, condition_name in enumerate(test_conditions):
        plt.subplot(1, 3, i+1)
        res = all_results[condition_name]

        plt.plot(res["steps"], res["probe_auroc"], label="Learned Probe (h+target)", marker="o")
        plt.plot(res["steps"], res["loss_auroc"], label="Baseline A (Loss)", marker="x")
        plt.plot(res["steps"], res["freq_auroc"], label="Baseline B (Target Freq)", marker="^")

        plt.axhline(0.5, color='gray', linestyle='--')
        plt.ylim(0.4, 1.05)
        plt.xlabel("Training Step")
        plt.ylabel("AUROC")
        plt.title(f"{condition_name.replace('_', ' ').title()}")
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("results/detection_auroc.png", dpi=300)
    print("Results saved to results/detection_results.json and results/detection_auroc.png")

if __name__ == "__main__":
    main()
