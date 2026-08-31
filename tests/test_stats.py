import math
from grokkit.stats import compute_weight_norm_stats
from grokkit.transplant import aggregate_transplant_results

def test_weight_norm_stats():
    runs = [
        {
            "condition": "cond1",
            "history": [
                {"step": 1, "weight_norm": 10.0},
                {"step": 2, "weight_norm": 20.0},
                {"step": 3, "weight_norm": 5.0}
            ]
        },
        {
            "condition": "cond1",
            "history": [
                {"step": 1, "weight_norm": 10.0},
                {"step": 2, "weight_norm": 10.0},
                {"step": 3, "weight_norm": 5.0}
            ]
        }
    ]

    stats = compute_weight_norm_stats(runs)
    assert "cond1" in stats
    # peaks: 20, 10 -> mean peak = 15
    # finals: 5, 5 -> mean final = 5
    # drops: 15/20 = 0.75, 5/10 = 0.5 -> mean drop = 0.625
    assert math.isclose(stats["cond1"]["peak_norm_mean"], 15.0)
    assert math.isclose(stats["cond1"]["final_norm_mean"], 5.0)
    assert math.isclose(stats["cond1"]["drop_pct_mean"], 0.625)

def test_transplant_results():
    res = [
        {"layer": 1, "head": 1, "rescue_score": 0.8},
        {"layer": 1, "head": 1, "rescue_score": 0.6},
        {"layer": 2, "head": 2, "rescue_score": 0.9}
    ]

    agg = aggregate_transplant_results(res)
    assert math.isclose(agg["L1H1"], 0.7)
    assert math.isclose(agg["L2H2"], 0.9)
