import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.log_loader import load_results_json
from eval.early_features import compute_rolling_features
import itertools
from collections import defaultdict

def evaluate_predictor_crossing(
    feature_values: np.ndarray,
    steps: List[int],
    baseline_steps: int = 500,
    threshold_sigma: float = 3.0,
    direction: str = "up"
) -> int:
    """
    Returns the step at which the predictor crosses the baseline threshold.
    Returns -1 if it never crosses.
    """
    # find indices in the baseline window
    baseline_indices = [i for i, s in enumerate(steps) if s <= baseline_steps and not np.isnan(feature_values[i])]
    if not baseline_indices:
        return -1

    baseline_vals = feature_values[baseline_indices]
    mean = np.mean(baseline_vals)
    std = np.std(baseline_vals)
    if std == 0:
        std = 1e-6 # fallback to avoid zero variance issues

    for i, (step, val) in enumerate(zip(steps, feature_values)):
        if step <= baseline_steps or np.isnan(val):
            continue

        if direction == "up" and val > mean + threshold_sigma * std:
            return step
        elif direction == "down" and val < mean - threshold_sigma * std:
            return step

    return -1

def evaluate_framework():
    grid_dir = Path("results/grid")
    if not grid_dir.exists():
        print(f"Warning: {grid_dir} does not exist. Skipping evaluation.")
        return None

    # Load all runs
    runs = []
    for p in grid_dir.rglob("results.json"):
        try:
            data = load_results_json(p.parent)
            cfg = data.get("config", {})
            runs.append({
                "path": p.parent,
                "severity": cfg.get("collapse_severity", 0.0),
                "level": cfg.get("collapse_level", 0.0),
                "grokked": data.get("grokked", False),
                "grokking_step": data.get("grokking_step", -1),
                "history": data.get("history", [])
            })
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if not runs:
        print("No valid runs found.")
        return None

    print(f"Loaded {len(runs)} runs.")

    predictors = {
        "loss_gap": "up",
        "weight_norm_slope": "down",
        "effective_rank": "up",
        "test_acc_curvature": "up",
        "activation_sparsity": "up",
        "gradient_norm": "up"
    }

    results = defaultdict(list)
    severities = sorted(list(set([r["severity"] for r in runs])))

    for val_severity in severities:
        val_runs = [r for r in runs if r["severity"] == val_severity]

        for r in val_runs:
            features = compute_rolling_features(r["history"], window_size=5)
            steps = features["step"]
            actual_grok = r["grokked"]
            actual_step = r["grokking_step"]

            run_result = {
                "severity": r["severity"],
                "grokked": actual_grok,
                "grokking_step": actual_step,
            }

            for p_name, direction in predictors.items():
                if p_name not in features:
                    continue
                crossing_step = evaluate_predictor_crossing(
                    features[p_name], steps, baseline_steps=500, threshold_sigma=3.0, direction=direction
                )

                if crossing_step > -1:
                    is_early = (actual_grok and crossing_step <= actual_step * 0.5)
                    run_result[p_name] = {
                        "crossed": True,
                        "step": crossing_step,
                        "valid_early_warning": is_early
                    }
                else:
                    run_result[p_name] = {
                        "crossed": False,
                        "step": -1,
                        "valid_early_warning": False
                    }
            results[val_severity].append(run_result)

    return results, predictors

def generate_report(results, predictors, output_path: Path):
    with open(output_path, "w") as f:
        f.write("# Early-Warning Predictors for Grokking\n\n")
        f.write("Evaluation based on leave-one-severity-out validation.\n\n")

        f.write("## Predictor Rankings\n\n")
        f.write("| Predictor | Lead Time (steps) | Precision | Recall | FPR (on never-grok) |\n")
        f.write("|-----------|-------------------|-----------|--------|---------------------|\n")

        best_combo = None
        best_f1 = -1

        # We also evaluate pairs (OR combinator) to find the best combo
        pairs = list(itertools.combinations(predictors.keys(), 2))
        combo_stats = {}

        all_runs = []
        for sev, runs in results.items():
            all_runs.extend(runs)

        def compute_stats(name, eval_func):
            tp, fp, fn, tn = 0, 0, 0, 0
            lead_times = []

            for r in all_runs:
                actual = r["grokked"]
                crossed, is_valid, step = eval_func(r)
                if actual:
                    if is_valid:
                        tp += 1
                        lead_times.append(r["grokking_step"] - step)
                    else:
                        fn += 1
                else:
                    if crossed:
                        fp += 1
                    else:
                        tn += 1

            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
            avg_lead = np.mean(lead_times) if lead_times else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
            return avg_lead, prec, rec, fpr, f1

        for p_name in predictors.keys():
            def single_eval(r, p=p_name):
                return r[p]["crossed"], r[p]["valid_early_warning"], r[p]["step"]

            avg_lead, prec, rec, fpr, f1 = compute_stats(p_name, single_eval)
            f.write(f"| {p_name} | {avg_lead:.1f} | {prec:.3f} | {rec:.3f} | {fpr:.3f} |\n")

            if f1 > best_f1:
                best_f1 = f1
                best_combo = p_name

        f.write("\n## Best Early-Warning Combo\n\n")
        # Find best pair
        for p1, p2 in pairs:
            def pair_eval(r, p1=p1, p2=p2):
                c1 = r[p1]["crossed"]
                c2 = r[p2]["crossed"]
                crossed = c1 or c2
                v1 = r[p1]["valid_early_warning"]
                v2 = r[p2]["valid_early_warning"]
                is_valid = v1 or v2
                step = -1
                if v1 and v2: step = min(r[p1]["step"], r[p2]["step"])
                elif v1: step = r[p1]["step"]
                elif v2: step = r[p2]["step"]
                return crossed, is_valid, step

            avg_lead, prec, rec, fpr, f1 = compute_stats(f"{p1} OR {p2}", pair_eval)
            combo_stats[f"{p1} OR {p2}"] = (avg_lead, prec, rec, fpr, f1)

            if f1 > best_f1:
                best_f1 = f1
                best_combo = f"{p1} OR {p2}"

        if best_combo:
            if best_combo in predictors.keys():
                avg_lead, prec, rec, fpr, f1 = compute_stats(best_combo, lambda r: (r[best_combo]["crossed"], r[best_combo]["valid_early_warning"], r[best_combo]["step"]))
            else:
                avg_lead, prec, rec, fpr, f1 = combo_stats[best_combo]

            f.write(f"**Recommended Combo**: `{best_combo}`\n")
            f.write(f"- Lead Time: {avg_lead:.1f} steps\n")
            f.write(f"- Precision: {prec:.3f}\n")
            f.write(f"- Recall: {rec:.3f}\n")
            f.write(f"- FPR: {fpr:.3f}\n")

        f.write("\n## Null-Hypothesis Check (Shuffled Labels)\n\n")
        f.write("To ensure predictors are better than chance, we shuffle the 'grokked' labels across runs.\n\n")

        np.random.seed(42)
        shuffled_labels = [r["grokked"] for r in all_runs]
        np.random.shuffle(shuffled_labels)

        f.write("| Predictor | Shuffled Precision | Shuffled Recall |\n")
        f.write("|-----------|--------------------|-----------------|\n")
        for p_name in predictors.keys():
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for r, shuf_label in zip(all_runs, shuffled_labels):
                pred = r.get(p_name, {})
                valid = pred.get("crossed", False)

                if shuf_label:
                    if valid: true_positives += 1
                    else: false_negatives += 1
                else:
                    if valid: false_positives += 1

            shuf_prec = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            shuf_rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f.write(f"| {p_name} | {shuf_prec:.3f} | {shuf_rec:.3f} |\n")

if __name__ == "__main__":
    res = evaluate_framework()
    if res:
        results, predictors = res
        out_dir = Path("analysis")
        out_dir.mkdir(exist_ok=True)
        generate_report(results, predictors, out_dir / "EARLY_WARNING.md")
        print(f"Report generated at {out_dir / 'EARLY_WARNING.md'}")
