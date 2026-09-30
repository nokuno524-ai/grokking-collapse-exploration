with open("src/prediction/outcome_predictor.py", "r") as f:
    code = f.read()
code = code.replace(
    'features["tt_gap_slope"] = tt_slope if tt_slope is not None else float(\'nan\')',
    'features["tt_gap_slope"] = tt_slope if tt_slope is not None else float(\'nan\')\n    from .early_warning import compute_weight_norm_trajectory\n    wn = compute_weight_norm_trajectory(early_history)\n    features["weight_norm_slope"] = (wn[-1] - wn[0]) if len(wn) > 1 else float("nan")'
)
code = code.replace(
    'X = np.array([\n        [d["tt_gap_slope"], d["eff_rank_last"], d["attn_entropy_slope"]]',
    'X = np.array([\n        [d["tt_gap_slope"], d["eff_rank_last"], d["attn_entropy_slope"], d.get("weight_norm_slope", float("nan"))]'
)
with open("src/prediction/outcome_predictor.py", "w") as f:
    f.write(code)
