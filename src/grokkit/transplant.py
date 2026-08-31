from typing import Dict, List, Any

def aggregate_transplant_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates causal circuit / transplant rescue results.
    """
    agg = {}
    # Example logic: assume list of { "layer": 1, "head": 2, "rescue_score": 0.8 }
    for res in results:
        layer = res.get("layer")
        head = res.get("head")
        score = res.get("rescue_score")
        if layer is not None and head is not None and score is not None:
            key = f"L{layer}H{head}"
            if key not in agg:
                agg[key] = []
            agg[key].append(score)

    summary = {}
    for k, v in agg.items():
        summary[k] = sum(v) / len(v)
    return summary
