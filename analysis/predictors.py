import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import analysis.linkage as link

def compute_early_warning_metrics(run_data, pre_grok_steps=1000):
    history = run_data.get('history', [])
    if not history:
        return None

    steps = np.array([h['step'] for h in history])
    valid_indices = steps <= pre_grok_steps

    if not np.any(valid_indices):
        return None

    early_steps = steps[valid_indices]
    early_wn = np.array([h['weight_norm'] for h in history])[valid_indices]
    early_train_loss = np.array([h['train_loss'] for h in history])[valid_indices]
    early_test_acc = np.array([h['test_acc'] for h in history])[valid_indices]

    if len(early_steps) < 3:
        return None

    slope, _ = np.polyfit(early_steps, early_wn, 1)
    poly_coeffs = np.polyfit(early_steps, early_train_loss, 2)
    curvature = poly_coeffs[0]
    acc_variance = np.var(early_test_acc)

    return {
        'wn_slope': slope,
        'loss_curvature': curvature,
        'acc_variance': acc_variance,
    }

def train_predictor_and_evaluate(runs_metrics, early_metrics_list):
    X = []
    y = []

    for metrics, early in zip(runs_metrics, early_metrics_list):
        if early is not None:
            features = [
                early['wn_slope'],
                early['loss_curvature'],
                early['acc_variance']
            ]
            X.append(features)
            y.append(1 if metrics['grok_success'] else 0)

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        return None, None, None, None, None

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std

    clf = LogisticRegression(random_state=42, class_weight='balanced')
    clf.fit(X_scaled, y)

    y_scores = clf.predict_proba(X_scaled)[:, 1]
    auroc = roc_auc_score(y, y_scores)

    # Compute Bootstrap CI for AUROC
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        score = roc_auc_score(y[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    ci = (ci_lower, ci_upper)

    return clf, auroc, ci, y, y_scores
