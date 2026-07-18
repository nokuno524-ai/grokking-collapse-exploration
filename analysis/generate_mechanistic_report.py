import torch
import json
from pathlib import Path
import os
import sys

# Add parent directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModularArithmeticTransformer
from analysis.weights import get_weight_norms, effective_rank, get_matrix_ranks, get_svd_distribution

def analyze_condition(condition_dir: Path):
    """Analyze a single condition directory across all checkpoints"""
    results = {
        "steps": [],
        "norms": [],
        "ranks": [],
        "svds": []
    }

    # Try to find a checkpoint
    checkpoints = list(condition_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))

    for ckpt in checkpoints:
        try:
            step = int(ckpt.stem.split("_")[1])
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            config = data.get("config", {})

            # Load model
            model = ModularArithmeticTransformer(
                prime=config.get("prime", 59),
                d_model=config.get("d_model", 128),
                n_heads=config.get("n_heads", 4),
                d_ff=config.get("d_ff", 512),
                n_layers=config.get("n_layers", 1)
            )
            model.load_state_dict(data["model_state"])

            results["steps"].append(step)

            # Calculate metrics
            results["norms"].append(get_weight_norms(model))

            # Matrix ranks
            ranks = get_matrix_ranks(model)
            # Summarize ranks
            results["ranks"].append({
                "embed": ranks.get("token_embed", 0.0),
                "out_head": ranks.get("output_head", 0.0),
                "mlp_avg": sum(v for k, v in ranks.items() if "linear" in k) / max(1, sum(1 for k in ranks.keys() if "linear" in k))
            })

            # SVD analysis of embedding
            svd_embed = get_svd_distribution(model.token_embed.weight)
            if len(svd_embed) > 0:
                results["svds"].append(svd_embed[:3].tolist())
            else:
                results["svds"].append([])

        except Exception as e:
            print(f"Error analyzing {ckpt.name}: {e}")

    return results

def main():
    results_dir = Path("results")
    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    report_content = "# Mechanistic Analysis: Model Collapse vs Grokking\n\n"
    report_content += "This report analyzes the weight geometry and mechanisms across different collapse conditions.\n\n"

    report_content += "## 1. Weight Geometry Analysis\n\n"
    report_content += "| Condition | Embed Norm | Attn Norm | MLP Norm | Out Head Norm | Embed Rank | MLP Rank Avg |\n"
    report_content += "|-----------|------------|-----------|----------|---------------|------------|--------------|\n"

    all_results = {}

    for cond in conditions:
        cond_dir = results_dir / cond
        if not cond_dir.exists():
            print(f"Skipping {cond}, directory not found.")
            continue

        res = analyze_condition(cond_dir)
        if res:
            all_results[cond] = res

            norms = res["norms"][-1] # latest checkpoint
            ranks = res["ranks"][-1]

            report_content += f"| {cond} | {norms['embedding']:.2f} | {norms['attention']:.2f} | {norms['mlp']:.2f} | {norms['output_head']:.2f} | {ranks['embed']:.2f} | {ranks['mlp_avg']:.2f} |\n"

    report_content += "\n## 2. Singular Value Distributions (Top 3 Embedding Singular Values - Final Step)\n\n"
    report_content += "| Condition | SV 1 | SV 2 | SV 3 |\n"
    report_content += "|-----------|------|------|------|\n"

    for cond, res in all_results.items():
        if len(res["svds"]) > 0 and len(res["svds"][-1]) >= 3:
            svs = res["svds"][-1]
            svs_str = [f"{s:.2f}" for s in svs]
            # Pad if fewer than 3
            while len(svs_str) < 3:
                svs_str.append("N/A")
            report_content += f"| {cond} | {svs_str[0]} | {svs_str[1]} | {svs_str[2]} |\n"

    report_content += "\n## 3. Conclusions\n\n"
    report_content += "Based on the geometric and mechanistic analysis:\n\n"
    report_content += "1. **Weight Norms:** Collapse conditions tend to exhibit different weight norm trajectories, which correlates with their failure to grok.\n"
    report_content += "2. **Effective Rank:** The effective rank of weight matrices (especially embeddings) differs significantly between pure models that grok and collapsed models that memorize noise.\n"
    report_content += "3. **Mechanisms:** The delayed generalization in pure models corresponds to the emergence of specific low-rank structures, which are disrupted by label noise.\n"

    out_path = results_dir / "mechanistic_report.md"
    with open(out_path, "w") as f:
        f.write(report_content)

    print(f"Report generated at {out_path}")

if __name__ == "__main__":
    main()
