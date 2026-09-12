"""
Multiplicity harness for evaluating grokking claims across an analytic grid.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

from .grok_detector.detectors import fixed_threshold_detector, max_2nd_derivative_detector, plateau_detector
from .grok_detector.stats import spearman_correlation, bootstrap_ci, benjamini_hochberg

# 0: pure, 1: low, 2: medium, 3: high, 4: severe
SEVERITY_MAP = {
    "pure": 0,
    "low_collapse": 1,
    "medium_collapse": 2,
    "high_collapse": 3,
    "severe_collapse": 4
}

def load_run_history(run_dir: Path) -> List[Dict]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return []
    try:
        with open(results_path) as f:
            data = json.load(f)
            return data.get("history", [])
    except Exception:
        return []

def get_weight_norm_reduction(history: List[Dict], window: int) -> float:
    if not history:
        return 0.0
    weight_norms = [e.get("weight_norm", 0.0) for e in history]
    if window > 1:
        # Simple smooth
        kernel = np.ones(window) / window
        if len(weight_norms) >= window:
            weight_norms = np.convolve(weight_norms, kernel, mode='valid').tolist()
    if not weight_norms:
        return 0.0
    return max(weight_norms) - min(weight_norms)


def run_harness(results_dir: Path) -> Dict[str, Any]:
    """
    Run the multiplicity harness across all analytic regimes.
    Returns the results dictionary mapping regime names to metrics and claim verdicts.
    """
    # Define grid
    detectors = {
        "fixed": lambda steps, vals, window, t: fixed_threshold_detector(steps, vals, t, window),
        "max_2nd_der": lambda steps, vals, window, t: max_2nd_derivative_detector(steps, vals, window),
        "plateau": lambda steps, vals, window, t: plateau_detector(steps, vals, window, patience=5, plateau_threshold=t)
    }
    thresholds = [0.8, 0.9, 0.95]
    windows = [1, 5, 10]

    # Locate seeds
    multi_seed_dir = results_dir / "multi_seed"
    if not multi_seed_dir.exists():
        return {}

    seeds = [p for p in multi_seed_dir.iterdir() if p.is_dir() and p.name.isdigit()]

    # Parse data first to avoid re-reading files
    parsed_data = {}
    for seed_dir in seeds:
        seed = seed_dir.name
        parsed_data[seed] = {}
        for cond, sev in SEVERITY_MAP.items():
            cond_dir = seed_dir / cond
            history = load_run_history(cond_dir)
            if history:
                steps = [e["step"] for e in history]
                accs = [e.get("test_acc", 0.0) for e in history]
                parsed_data[seed][cond] = {
                    "history": history,
                    "steps": steps,
                    "accs": accs,
                    "severity": sev
                }

    regimes = {}
    claim1_p_values = []
    claim2_p_values = []

    for det_name, det_func in detectors.items():
        for t in thresholds:
            for w in windows:
                regime_name = f"{det_name}_t{t}_w{w}"

                # We collect data for this regime
                pure_grok_steps = []
                med_severe_grokked = False

                severities_for_corr = []
                wn_reductions_for_corr = []

                for seed, conds in parsed_data.items():
                    for cond_name, data in conds.items():
                        # Grok step detection
                        grok_step = det_func(data["steps"], data["accs"], w, t)

                        if cond_name == "pure" and grok_step is not None:
                            pure_grok_steps.append(grok_step)
                        if cond_name in ["medium_collapse", "severe_collapse"] and grok_step is not None:
                            med_severe_grokked = True

                        # Weight norm reduction
                        wn_red = get_weight_norm_reduction(data["history"], w)
                        severities_for_corr.append(data["severity"])
                        wn_reductions_for_corr.append(wn_red)

                # Claim 1 Evaluation: pure groks at < 2000, medium/severe never grok
                claim1_survives = False
                if not med_severe_grokked and len(pure_grok_steps) > 0:
                    avg_pure_grok = np.mean(pure_grok_steps)
                    if avg_pure_grok < 2000:
                        claim1_survives = True

                # Claim 2 Evaluation: weight-norm reduction correlates with severity (negative correlation)
                # i.e., more severe collapse -> less weight norm reduction
                claim2_survives = False
                corr, p_val = spearman_correlation(np.array(severities_for_corr), np.array(wn_reductions_for_corr))
                if not np.isnan(corr) and corr < 0: # Check if correlation is negative
                    claim2_p_values.append((regime_name, p_val, corr))
                else:
                    claim2_p_values.append((regime_name, 1.0, corr)) # High p-value if wrong direction or nan

                regimes[regime_name] = {
                    "claim1_survives": claim1_survives,
                    "pure_grok_mean": np.mean(pure_grok_steps) if pure_grok_steps else None,
                    "med_severe_grokked": med_severe_grokked,
                    "claim2_corr": corr,
                    "claim2_raw_p": p_val
                }

    # Multiple comparison correction for Claim 2
    raw_ps = [p for _, p, _ in claim2_p_values]
    rejects = benjamini_hochberg(raw_ps, alpha=0.05)

    for (regime_name, raw_p, corr), reject in zip(claim2_p_values, rejects):
        # Claim 2 survives if it's significant AND negative
        regimes[regime_name]["claim2_survives"] = bool(reject and corr < 0)
        regimes[regime_name]["claim2_corrected_sig"] = bool(reject)

    return regimes


def summarize_robustness(regimes: Dict[str, Any]) -> str:
    """Generate a markdown summary of the robustness across regimes."""
    total_regimes = len(regimes)
    if total_regimes == 0:
        return "No regimes evaluated (missing data)."

    claim1_pass = sum(1 for r in regimes.values() if r["claim1_survives"])
    claim2_pass = sum(1 for r in regimes.values() if r["claim2_survives"])

    report = []
    report.append("## Robustness of headline claims\n")
    report.append(f"We stress-tested the grokking cliffs and weight-norm reduction claims across {total_regimes} defensible analytic regimes (varying smoothing windows, threshold metrics, and accuracy targets).\n")

    report.append("### Claim 1: Pure runs grok early (< 2000 steps), while Medium/Severe collapse never grok.")
    report.append(f"- **Survival Rate**: {claim1_pass}/{total_regimes} ({claim1_pass/total_regimes*100:.1f}%) regimes.")

    report.append("### Claim 2: Weight-norm reduction strictly correlates (negatively) with collapse severity.")
    report.append(f"- **Survival Rate**: {claim2_pass}/{total_regimes} ({claim2_pass/total_regimes*100:.1f}%) regimes (Benjamini-Hochberg corrected, alpha=0.05).")

    report.append("\n### Detailed Regime Breakdown\n")
    report.append("| Regime | Claim 1 Survives | Claim 2 Survives | Claim 2 Corr |")
    report.append("|--------|------------------|------------------|--------------|")

    for name, metrics in sorted(regimes.items()):
        c1 = "✅" if metrics["claim1_survives"] else "❌"
        c2 = "✅" if metrics["claim2_survives"] else "❌"
        corr_str = f"{metrics['claim2_corr']:.3f}" if not np.isnan(metrics['claim2_corr']) else "NaN"
        report.append(f"| {name} | {c1} | {c2} | {corr_str} |")

    return "\n".join(report)

if __name__ == "__main__":
    results_dir = Path("results")
    regimes = run_harness(results_dir)
    print(summarize_robustness(regimes))
