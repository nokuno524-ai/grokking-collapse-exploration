import math
from typing import Dict, List, Any, Optional

def compute_weight_norm_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes weight norm statistics across a set of runs.
    """
    stats = {}
    for r in runs:
        cond = r.get("condition", "unknown")
        history = r.get("history", [])

        norms = [e["weight_norm"] for e in history if "weight_norm" in e and not math.isnan(e["weight_norm"])]
        if norms:
            peak = max(norms)
            final = norms[-1]
            drop = (peak - final) / peak if peak > 0 else 0

            if cond not in stats:
                stats[cond] = {"peaks": [], "finals": [], "drops": []}
            stats[cond]["peaks"].append(peak)
            stats[cond]["finals"].append(final)
            stats[cond]["drops"].append(drop)

    summary = {}
    for cond, data in stats.items():
        summary[cond] = {
            "peak_norm_mean": sum(data["peaks"]) / len(data["peaks"]) if data["peaks"] else float('nan'),
            "final_norm_mean": sum(data["finals"]) / len(data["finals"]) if data["finals"] else float('nan'),
            "drop_pct_mean": sum(data["drops"]) / len(data["drops"]) if data["drops"] else float('nan'),
        }
    return summary
