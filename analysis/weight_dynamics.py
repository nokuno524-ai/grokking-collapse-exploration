import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from scipy import stats

def compute_effective_rank(W):
    """SVD entropy-based effective rank."""
    if W.dim() > 2:
        W = W.reshape(W.shape[0], -1)

    # Use torch.linalg.svdvals as torch.svd is deprecated
    s = torch.linalg.svdvals(W)

    # Check for near-zero sum to prevent NaN
    s_sum = s.sum()
    if s_sum < 1e-10:
        return 0.0

    s_norm = s / s_sum
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    return torch.exp(entropy).item()

def analyze_weight_dynamics(results_dir, output_dir):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = {
        "pure": 0.0,
        "low_collapse": 0.3,
        "medium_collapse": 0.5,
        "high_collapse": 0.7,
        "severe_collapse": 0.9
    }

    # Store summary stats for scatter plots
    summary = []

    for condition, severity in conditions.items():
        cond_dir = results_path / condition
        if not cond_dir.exists():
            continue

        print(f"Processing condition: {condition}")

        checkpoints = list(cond_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            continue

        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except:
                return -1

        checkpoints.sort(key=get_step)

        steps = []
        metrics = {
            "norm_embed": [],
            "norm_attn_in": [],
            "norm_attn_out": [],
            "norm_mlp1": [],
            "norm_mlp2": [],
            "norm_total": [],
            "rank_embed": [],
            "rank_attn_out": []
        }

        for ckpt_path in checkpoints:
            step = get_step(ckpt_path)

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt['model_state']

                # Strip module prefix if exists
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                w_embed = state_dict['token_embed.weight']
                w_attn_in = state_dict['transformer.layers.0.self_attn.in_proj_weight']
                w_attn_out = state_dict['transformer.layers.0.self_attn.out_proj.weight']
                w_mlp1 = state_dict['transformer.layers.0.linear1.weight']
                w_mlp2 = state_dict['transformer.layers.0.linear2.weight']

                metrics["norm_embed"].append(torch.norm(w_embed).item())
                metrics["norm_attn_in"].append(torch.norm(w_attn_in).item())
                metrics["norm_attn_out"].append(torch.norm(w_attn_out).item())
                metrics["norm_mlp1"].append(torch.norm(w_mlp1).item())
                metrics["norm_mlp2"].append(torch.norm(w_mlp2).item())

                total_norm = np.sqrt(sum(torch.norm(w).item()**2 for w in state_dict.values() if w.dtype == torch.float32))
                metrics["norm_total"].append(total_norm)

                metrics["rank_embed"].append(compute_effective_rank(w_embed))
                metrics["rank_attn_out"].append(compute_effective_rank(w_attn_out))

                steps.append(step)
            except Exception as e:
                print(f"  Failed loading {ckpt_path}: {e}")

        if not steps:
            continue

        steps = np.array(steps)

        # Plot norm trajectories
        plt.figure(figsize=(10, 6))
        plt.plot(steps, metrics["norm_embed"], label='Token Embed', marker='.')
        plt.plot(steps, metrics["norm_attn_in"], label='Attn In Proj', marker='.')
        plt.plot(steps, metrics["norm_attn_out"], label='Attn Out Proj', marker='.')
        plt.plot(steps, metrics["norm_mlp1"], label='MLP 1', marker='.')
        plt.plot(steps, metrics["norm_mlp2"], label='MLP 2', marker='.')

        plt.title(f'Weight Norm Trajectories - {condition}')
        plt.xlabel('Training Step')
        plt.ylabel('L2 Norm')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out_path / f'norm_traj_{condition}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot rank trajectories
        plt.figure(figsize=(10, 6))
        plt.plot(steps, metrics["rank_embed"], label='Token Embed Rank', marker='.')
        plt.plot(steps, metrics["rank_attn_out"], label='Attn Out Proj Rank', marker='.')

        plt.title(f'Effective Rank Trajectories - {condition}')
        plt.xlabel('Training Step')
        plt.ylabel('Effective Rank (SVD Entropy)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out_path / f'rank_traj_{condition}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate overall norm reduction
        norms = metrics["norm_total"]
        if norms:
            peak_norm = max(norms)
            final_norm = norms[-1]
            reduction = (peak_norm - final_norm) / peak_norm
            summary.append((condition, severity, reduction))

    # Compute correlation and CI for norm reduction vs collapse severity
    if len(summary) >= 3:
        severities = np.array([x[1] for x in summary])
        reductions = np.array([x[2] for x in summary])

        r, p_val = stats.pearsonr(severities, reductions)

        # Bootstrap CI
        n_boot = 1000
        boot_rs = []
        np.random.seed(42)
        n = len(severities)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            # Need at least 2 distinct values to compute correlation
            if len(np.unique(severities[idx])) > 1 and len(np.unique(reductions[idx])) > 1:
                br, _ = stats.pearsonr(severities[idx], reductions[idx])
                boot_rs.append(br)

        if boot_rs:
            ci_low = np.percentile(boot_rs, 2.5)
            ci_high = np.percentile(boot_rs, 97.5)
        else:
            ci_low, ci_high = np.nan, np.nan

        print(f"Norm Reduction vs Severity: r = {r:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

        plt.figure(figsize=(8, 6))
        plt.scatter(severities, reductions, c='blue', s=100)

        # Add labels
        for cond, sev, red in summary:
            plt.annotate(cond, (sev, red), xytext=(5, 5), textcoords='offset points')

        # Add trend line
        z = np.polyfit(severities, reductions, 1)
        p = np.poly1d(z)
        plt.plot(severities, p(severities), "r--", alpha=0.8)

        plt.title('Weight Norm Reduction vs Collapse Severity')
        plt.xlabel('Collapse Severity')
        plt.ylabel('Weight Norm Reduction ((Peak - Final) / Peak)')
        plt.grid(alpha=0.3)
        plt.savefig(out_path / 'norm_reduction_vs_severity.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analyze_weight_dynamics("results", "analysis/weights")
    print("Weight dynamics analysis complete.")
