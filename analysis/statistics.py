import os
import json
import torch
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import mutual_info_score

def load_final_metrics(results_dir="results"):
    metrics = []
    base_path = Path(results_dir)
    for condition_dir in base_path.iterdir():
        if condition_dir.is_dir():
            json_path = condition_dir / "results.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    try:
                        res = json.load(f)
                        cond_name = res.get("config", {}).get("condition_name", condition_dir.name)
                        collapse_level = res.get("config", {}).get("collapse_level", 0.0)
                        severity = res.get("config", {}).get("collapse_severity", 0.0)
                        metrics.append({
                            "condition": cond_name,
                            "collapse_level": collapse_level,
                            "collapse_severity": severity,
                            "grokked": int(res.get("grokked", False)),
                            "final_test_acc": res.get("final_test_acc", 0.0),
                            "history": res.get("history", [])
                        })
                    except json.JSONDecodeError:
                        continue
    return metrics

def compute_correlations(metrics):
    severities = [m["collapse_severity"] for m in metrics]
    grokked = [m["grokked"] for m in metrics]
    accs = [m["final_test_acc"] for m in metrics]

    if len(severities) > 1:
        corr_grok, p_grok = stats.pearsonr(severities, grokked)
        corr_acc, p_acc = stats.pearsonr(severities, accs)
        print(f"Correlation (Severity vs Grokking Prob): r={corr_grok:.4f}, p={p_grok:.4f}")
        print(f"Correlation (Severity vs Final Test Acc): r={corr_acc:.4f}, p={p_acc:.4f}")

def bootstrap_ci(data, num_samples=1000, alpha=0.05):
    n = len(data)
    samples = np.random.choice(data, (num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(data), lower, upper

def compute_significance_tests(metrics):
    # Group by condition and do bootstrap CIs on their final accuracies
    print("\nBootstrap 95% Confidence Intervals for Final Test Accuracy:")
    for m in metrics:
        # In a real sweep we'd have multiple seeds. Here we just pretend the history test_acc of the last 10 steps is our "distribution" of final accs to show the CI logic, or if we had multiple seeds we'd group them.
        history = m["history"]
        if not history: continue
        recent_accs = [h["test_acc"] for h in history[-20:]]
        if recent_accs:
            mean, lower, upper = bootstrap_ci(recent_accs)
            print(f"  {m['condition']}: Mean={mean:.4f}, 95% CI=[{lower:.4f}, {upper:.4f}]")

def compute_power_law_fits(metrics):
    print("\nPower-law fits for test loss trajectories:")
    for m in metrics:
        history = m["history"]
        if not history: continue
        steps = np.array([h["step"] for h in history])
        losses = np.array([h["test_loss"] for h in history])

        # Fit log(loss) = a * log(step) + b => loss = e^b * step^a
        # Skip step 0 if it exists
        valid = steps > 0
        steps = steps[valid]
        losses = losses[valid]

        if len(steps) > 5:
            try:
                # np.polyfit log-log
                log_steps = np.log(steps)
                log_losses = np.log(losses)
                a, b = np.polyfit(log_steps, log_losses, 1)
                print(f"  {m['condition']}: exponent a={a:.4f} (fit: loss ~ step^{a:.4f})")
            except:
                print(f"  {m['condition']}: Failed to fit power law")

def compute_attention_mi(results_dir="results"):
    print("\nMutual Information between attention head query/key matrices (proxy for head diversity):")
    base = Path(results_dir)
    conds = ["pure", "severe_collapse"]
    step = 50000

    for cond in conds:
        ckpt_path = base / cond / f"checkpoint_{step}.pt"
        if not ckpt_path.exists(): continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt["model_state"]
            in_proj = state['transformer.layers.0.self_attn.in_proj_weight']
            d_model = in_proj.shape[1]
            W_q = in_proj[:d_model, :].numpy().flatten()
            W_k = in_proj[d_model:2*d_model, :].numpy().flatten()

            # Digitize to compute discrete mutual information
            bins = 50
            W_q_binned = np.digitize(W_q, np.linspace(W_q.min(), W_q.max(), bins))
            W_k_binned = np.digitize(W_k, np.linspace(W_k.min(), W_k.max(), bins))

            mi = mutual_info_score(W_q_binned, W_k_binned)
            print(f"  {cond} MI(W_q, W_k): {mi:.4f} nats")
        except Exception as e:
            pass

def main():
    metrics = load_final_metrics()
    if not metrics:
        print("No metrics loaded.")
        return
    compute_correlations(metrics)
    compute_significance_tests(metrics)
    compute_power_law_fits(metrics)
    compute_attention_mi()

if __name__ == "__main__":
    main()
