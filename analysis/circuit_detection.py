import os
import glob
import torch
import numpy as np
import pandas as pd
import json

def load_results_json(filepath: str) -> dict:
    """Load results.json to get grokking information."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_weights(ckpt_path: str, device: str = 'cpu') -> dict:
    """Extract weight matrices from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    return ckpt['model_state']

def compute_effective_rank(w: torch.Tensor) -> float:
    """Compute effective rank using Shannon entropy of normalized singular values."""
    if len(w.shape) > 2:
        w = w.view(w.size(0), -1)
    elif len(w.shape) < 2:
        return 0.0

    s = torch.linalg.svdvals(w)
    s_norm = s / (s.sum() + 1e-10)
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    return torch.exp(entropy).item()

def compute_participation_ratio(w: torch.Tensor) -> float:
    """Compute participation ratio: (sum(s^2))^2 / sum(s^4)."""
    if len(w.shape) > 2:
        w = w.view(w.size(0), -1)
    elif len(w.shape) < 2:
        return 0.0

    s = torch.linalg.svdvals(w)
    s2 = s ** 2
    return ((s2.sum() ** 2) / (s2 ** 2).sum() + 1e-10).item()

def track_circuit_formation(base_dir: str = "results") -> pd.DataFrame:
    """Track circuit complexity metrics across training for all collapse levels."""
    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    target_layers = [
        'transformer.layers.0.self_attn.in_proj_weight',
        'transformer.layers.0.self_attn.out_proj.weight',
        'transformer.layers.0.linear1.weight',
        'transformer.layers.0.linear2.weight',
    ]

    data = []

    for condition in conditions:
        cond_dir = os.path.join(base_dir, condition)
        if not os.path.exists(cond_dir):
            continue

        results_info = load_results_json(os.path.join(cond_dir, "results.json"))
        grokking_step = results_info.get("grokking_step") or -1

        ckpts = glob.glob(os.path.join(cond_dir, "checkpoint_*.pt"))
        if not ckpts:
            continue

        for ckpt in ckpts:
            step = int(ckpt.split("checkpoint_")[1].split(".pt")[0])
            weights = extract_weights(ckpt)

            row = {
                "condition": condition,
                "step": step,
                "grokking_step": grokking_step,
                "is_post_grokking": step >= grokking_step if grokking_step != -1 else False
            }

            for layer in target_layers:
                if layer in weights:
                    w = weights[layer]
                    eff_rank = compute_effective_rank(w)
                    pr = compute_participation_ratio(w)

                    # Simplify column names
                    short_name = layer.split('.')[-2] if 'linear' in layer else layer.split('.')[-2]
                    if short_name == 'self_attn': short_name = 'in_proj'

                    row[f"{short_name}_eff_rank"] = eff_rank
                    row[f"{short_name}_pr"] = pr

            data.append(row)

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by=["condition", "step"]).reset_index(drop=True)
    return df

def analyze_circuit_correlations(df: pd.DataFrame, output_dir: str = "results/analysis_output"):
    """Analyze correlations between circuit metrics and grokking."""
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "circuit_metrics.csv"), index=False)

    # Calculate drops in effective rank
    summary_data = []

    for condition in df['condition'].unique():
        cond_df = df[df['condition'] == condition].sort_values('step')

        grokking_step = cond_df['grokking_step'].iloc[0]

        # Analyze in_proj effective rank
        if 'in_proj_eff_rank' in cond_df.columns:
            ranks = cond_df['in_proj_eff_rank'].values
            steps = cond_df['step'].values

            max_rank = ranks.max()
            min_rank = ranks.min()

            # Find when rank drops below 90% of max
            threshold = max_rank - 0.1 * (max_rank - min_rank)
            drop_indices = np.where(ranks < threshold)[0]

            rank_drop_step = steps[drop_indices[0]] if len(drop_indices) > 0 else -1

            summary_data.append({
                "condition": condition,
                "grokking_step": grokking_step,
                "rank_drop_step": rank_drop_step,
                "max_rank": max_rank,
                "min_rank": min_rank
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "circuit_correlation_summary.csv"), index=False)

    # Simple correlation analysis
    print("\n--- Circuit Formation Correlation ---")
    valid_groks = summary_df[summary_df['grokking_step'] != -1]
    if len(valid_groks) >= 2:
        corr = valid_groks['grokking_step'].corr(valid_groks['rank_drop_step'])
        print(f"Correlation between Rank Drop Step and Grokking Step: {corr:.3f}")
    else:
        print("Not enough data points that grokked to compute correlation.")

if __name__ == "__main__":
    df = track_circuit_formation()
    analyze_circuit_correlations(df)
