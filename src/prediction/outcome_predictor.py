"""
Outcome predictor using early-warning indicators to forecast grokking.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .early_warning import (
    compute_effective_rank_from_checkpoint,
    compute_train_test_gap_derivative,
    compute_attention_entropy_slope
)

def extract_features(run_dir: str) -> Dict[str, Any]:
    """
    Extract early-warning features from a run directory.
    Uses checkpoints up to 25% of the grokking step (or total steps).

    Args:
        run_dir (str): Path to run directory.

    Returns:
        dict: Features dictionary.
    """
    run_path = Path(run_dir)
    results_path = run_path / "results.json"

    if not results_path.exists():
        return {}

    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    history = data.get("history", [])
    config = data.get("config", {})
    grokked = data.get("grokked", False)
    grokking_step = data.get("grokking_step")

    if grokking_step is None:
        if history:
            grokking_step = history[-1].get("step", 50000)
        else:
            grokking_step = 50000

    cutoff_step = int(grokking_step * 0.25)

    # Filter history
    early_history = [e for e in history if e.get("step", 0) <= cutoff_step]

    features = {
        "grokked": grokked,
        "is_collapsed": config.get("collapse_level", 0.0) > 0.0,
        "collapse_level": config.get("collapse_level", 0.0),
        "collapse_severity": config.get("collapse_severity", 0.0)
    }

    # Train-test gap derivative
    tt_slope = compute_train_test_gap_derivative(early_history)
    features["tt_gap_slope"] = tt_slope if tt_slope is not None else float('nan')
    from .early_warning import compute_weight_norm_trajectory
    wn = compute_weight_norm_trajectory(early_history)
    features["weight_norm_slope"] = (wn[-1] - wn[0]) if len(wn) > 1 else float("nan")

    # Checkpoints
    ckpts = sorted(run_path.glob("checkpoint_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    early_ckpts = [p for p in ckpts if int(p.stem.split("_")[1]) <= cutoff_step]

    if early_ckpts:
        features["eff_rank_last"] = compute_effective_rank_from_checkpoint(str(early_ckpts[-1]))
        features["attn_entropy_slope"] = compute_attention_entropy_slope([str(p) for p in early_ckpts], config)
    else:
        features["eff_rank_last"] = float('nan')
        features["attn_entropy_slope"] = float('nan')

    if features["attn_entropy_slope"] is None:
        features["attn_entropy_slope"] = float('nan')

    return features

def train_and_evaluate(runs_data: List[Dict[str, Any]]):
    """
    Train and evaluate a logistic regression model using LOOCV.

    Args:
        runs_data (list of dict): Data extracted from runs.
    """
    # Filter out runs with NaN features
    valid_data = []
    for d in runs_data:
        if not d:
            continue
        if np.isnan(d.get("tt_gap_slope", float('nan'))) or \
           np.isnan(d.get("eff_rank_last", float('nan'))) or \
           np.isnan(d.get("attn_entropy_slope", float('nan'))):
            continue
        valid_data.append(d)

    if len(valid_data) < 2:
        print("Not enough valid runs to evaluate predictor.")
        return

    X = np.array([
        [d["tt_gap_slope"], d["eff_rank_last"], d["attn_entropy_slope"], d.get("weight_norm_slope", float("nan"))]
        for d in valid_data
    ])
    y = np.array([1 if d["grokked"] else 0 for d in valid_data])

    if len(np.unique(y)) < 2:
        print("Need both positive and negative examples.")
        return

    if not HAS_SKLEARN:
        print("scikit-learn not installed. Cannot run logistic regression.")
        return

    loo = LeaveOneOut()
    y_true = []
    y_pred_prob = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # If train fold lacks a class, fallback
        if len(np.unique(y_train)) < 2:
            pred = 0.5
        else:
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_train, y_train)
            pred = clf.predict_proba(X_test)[0, 1]

        y_true.append(y_test[0])
        y_pred_prob.append(pred)

    auc = roc_auc_score(y_true, y_pred_prob)

    # Calibration
    preds = np.array(y_pred_prob)
    trues = np.array(y_true)
    accuracy = np.mean((preds >= 0.5) == trues)

    # Performance on collapsed runs specifically
    collapsed_mask = np.array([d["is_collapsed"] for d in valid_data])
    if collapsed_mask.sum() > 0 and len(np.unique(trues[collapsed_mask])) > 1:
        col_auc = roc_auc_score(trues[collapsed_mask], preds[collapsed_mask])
    else:
        col_auc = float('nan')

    print(f"Outcome Predictor Evaluation (n={len(valid_data)} runs):")
    print(f"Overall AUC: {auc:.3f}")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Collapsed Runs AUC: {col_auc:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/grid")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    runs_data = []
    if results_path.exists():
        for run_dir in results_path.glob("**/seed_*"):
            if run_dir.is_dir() and (run_dir / "results.json").exists():
                feats = extract_features(str(run_dir))
                runs_data.append(feats)

    train_and_evaluate(runs_data)
