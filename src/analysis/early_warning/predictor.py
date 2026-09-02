"""
Driver for early warning prediction experiments.
Loads runs, computes signals at fractions of training, and fits predictors.
"""
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pathlib import Path
from .signals import compute_signals

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def log_loss(weights, X, y):
    z = np.dot(X, weights)
    p = sigmoid(z)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit_logistic_regression(X, y, lr=0.1, max_iter=1000):
    """Pure numpy logistic regression."""
    n_features = X.shape[1]
    w = np.zeros(n_features)
    for _ in range(max_iter):
        z = np.dot(X, w)
        p = sigmoid(z)
        gradient = np.dot(X.T, (p - y)) / len(y)
        w -= lr * gradient
    return w

def fit_linear_regression(X, y):
    """Simple linear regression using least squares."""
    # Add small ridge penalty for stability
    return np.linalg.solve(X.T @ X + 1e-5 * np.eye(X.shape[1]), X.T @ y)

def load_all_runs(base_dir: str) -> List[Dict]:
    """Loads all run results from multi_seed directory."""
    runs = []
    pattern = os.path.join(base_dir, "multi_seed", "*", "*", "results.json")
    for f in glob.glob(pattern):
        with open(f, 'r') as fp:
            data = json.load(fp)

        # Determine max step based on grokking step or max steps in history
        history = data.get('history', [])
        if not history:
            continue

        max_possible_step = history[-1]['step']
        grokked = data.get('grokked', False)
        grokking_step = data.get('grokking_step', None)

        # The 'event' step is either when it grokked, or the end of training
        event_step = grokking_step if grokked and grokking_step is not None else max_possible_step

        condition = Path(f).parent.name

        runs.append({
            'file': f,
            'history': history,
            'grokked': grokked,
            'grokking_step': grokking_step,
            'event_step': event_step,
            'condition': condition
        })
    return runs

def extract_dataset(runs: List[Dict], fraction: float, window_steps: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extracts features (X), binary targets (y_grok), continuous targets (y_step), and feature names
    at a specific fraction of the event step.
    """
    X_list = []
    y_grok_list = []
    y_step_list = []

    feature_names = ['train_loss_slope', 'weight_norm_slope', 'test_acc_var', 'delayed_gen_score']

    valid_runs = []
    for run in runs:
        cutoff_step = int(run['event_step'] * fraction)

        signals = compute_signals(run['history'], max_step=cutoff_step, window_steps=window_steps)

        # Only use runs where we can compute all our primary signals
        if not signals or any(np.isnan(signals.get(f, np.nan)) for f in feature_names):
            continue

        features = [signals[f] for f in feature_names]
        X_list.append(features)
        y_grok_list.append(1 if run['grokked'] else 0)
        y_step_list.append(run['event_step'])
        valid_runs.append(run)

    X = np.array(X_list)
    y_grok = np.array(y_grok_list)
    y_step = np.array(y_step_list)

    # Do not normalize here to avoid CV leakage.
    # Normalization will happen inside LOOCV.
    if len(X) > 0:
        # Add bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    return X, y_grok, y_step, ['bias'] + feature_names, valid_runs

def evaluate_loocv(X: np.ndarray, y: np.ndarray, task: str = 'classification'):
    """Leave-One-Out Cross-Validation with proper feature scaling."""
    n = len(X)
    predictions = np.zeros(n)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw = X[i:i+1]

        # Scale features based on training fold (ignore bias column 0)
        X_train = np.copy(X_train_raw)
        X_test = np.copy(X_test_raw)

        if X_train.shape[1] > 1:
            mean = np.mean(X_train[:, 1:], axis=0)
            std = np.std(X_train[:, 1:], axis=0)
            std[std == 0] = 1.0

            X_train[:, 1:] = (X_train[:, 1:] - mean) / std
            X_test[:, 1:] = (X_test[:, 1:] - mean) / std

        if task == 'classification':
            if len(np.unique(y_train)) < 2:
                predictions[i] = y_train[0]
            else:
                w = fit_logistic_regression(X_train, y_train)
                prob = sigmoid(np.dot(X_test, w))[0]
                predictions[i] = 1 if prob >= 0.5 else 0
        else: # regression
            w = fit_linear_regression(X_train, y_train)
            predictions[i] = np.dot(X_test, w)[0]

    if task == 'classification':
        accuracy = np.mean(predictions == y)
        return accuracy
    else:
        mae = np.mean(np.abs(predictions - y))
        return mae

def generate_report_and_plots(runs: List[Dict], fractions: List[float], out_dir: str):
    """Runs experiments and generates markdown report and plots."""
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for f in fractions:
        X, y_grok, y_step, feature_names, valid_runs = extract_dataset(runs, f)
        if len(X) < 5:
            print(f"Skipping fraction {f}, only {len(X)} valid samples.")
            continue

        acc = evaluate_loocv(X, y_grok, 'classification')
        grok_mask = y_grok == 1
        mae_grok = np.nan
        if np.sum(grok_mask) > 3: # Only evaluate step prediction on runs that actually grok
            mae_grok = evaluate_loocv(X[grok_mask], y_step[grok_mask], 'regression')

        results.append({
            'fraction': f,
            'samples': len(X),
            'grok_accuracy': acc,
            'grok_step_mae': mae_grok
        })

        # Generate plot for this fraction (boxplots of signals by grok status)
        fig, axes = plt.subplots(1, len(feature_names)-1, figsize=(15, 4))
        for i, feat in enumerate(feature_names[1:]): # skip bias
            ax = axes[i]
            grok_vals = X[grok_mask, i+1]
            non_grok_vals = X[~grok_mask, i+1]
            # Use tick_labels instead of labels for newer matplotlib
            ax.boxplot([grok_vals, non_grok_vals], tick_labels=['Grok', 'No Grok'])
            ax.set_title(feat)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'signals_f{f:.2f}.png'))
        plt.close()

    # Write report
    report_path = os.path.join(out_dir, 'grokking_early_warning.md')
    with open(report_path, 'w') as fp:
        fp.write("# Early Warning Signals for Grokking\n\n")
        fp.write("Can we predict whether a model will grok, and when, based on early training dynamics?\n\n")

        fp.write("## Prediction Accuracy vs Fraction of Training\n\n")
        fp.write("| Fraction of Training | Samples | Will-it-Grok Accuracy | Grok Step MAE |\n")
        fp.write("|---|---|---|---|\n")
        for res in results:
            fp.write(f"| {res['fraction']:.2f} | {res['samples']} | {res['grok_accuracy']:.1%} | {res['grok_step_mae']:.0f} |\n")

        fp.write("\n## Signals Separation\n\n")
        for f in fractions:
            fp.write(f"### At fraction {f:.2f}\n")
            fp.write(f"![Signals at f={f:.2f}](signals_f{f:.2f}.png)\n\n")

    return results

if __name__ == "__main__":
    runs = load_all_runs("results")
    print(f"Loaded {len(runs)} runs.")
    generate_report_and_plots(runs, [0.25, 0.5, 0.75], "analysis/early_warning")
