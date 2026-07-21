import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.stats as stats

def extract_early_features(history: List[Dict[str, Any]], early_steps: int = 1000) -> np.ndarray:
    """
    Extract features from early training history.
    Features: max slope of train loss, variance of gradients (if available), max train-test gap.
    """
    # Filter to early steps
    early_history = [h for h in history if h.get('step', 0) <= early_steps]
    if not early_history:
        # Fallback if step is not properly tracked, just take first N
        early_history = history[:min(len(history), 10)]

    train_losses = [h.get('train_loss', 0.0) for h in early_history]
    test_losses = [h.get('test_loss', 0.0) for h in early_history]
    grad_norms = [h.get('grad_norm', 0.0) for h in early_history]

    # Feature 1: Max slope of train loss (negative is good)
    if len(train_losses) > 1:
        loss_diffs = np.diff(train_losses)
        max_slope = np.min(loss_diffs) # Most negative
    else:
        max_slope = 0.0

    # Feature 2: Variance of gradients
    grad_var = np.var(grad_norms) if grad_norms else 0.0

    # Feature 3: Max train-test gap
    gaps = [test - train for test, train in zip(test_losses, train_losses)]
    max_gap = np.max(gaps) if gaps else 0.0

    return np.array([max_slope, grad_var, max_gap])

def train_logistic_regression(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """
    Train a simple logistic regression classifier to predict grokking.
    """
    model = LogisticRegression(random_state=42)

    # Handle single class case gracefully
    if len(np.unique(labels)) < 2:
        # Fake fit if only one class
        model.classes_ = np.unique(labels)
        model.coef_ = np.zeros((1, features.shape[1]))
        model.intercept_ = np.zeros((1,))
        return model

    model.fit(features, labels)
    return model

def evaluate_predictor(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """
    Evaluate the predictor and report accuracy and confidence interval.
    """
    if len(np.unique(y_test)) < 2 and not hasattr(model, 'coef_'):
        return 1.0, (1.0, 1.0)

    if not hasattr(model, 'coef_'):
        # Was not properly fitted
        preds = np.zeros_like(y_test)
    else:
        # Handle case where model only saw 1 class during training
        if len(model.classes_) == 1:
            preds = np.full(y_test.shape, model.classes_[0])
        else:
            preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    # CI for binomial proportion using normal approximation
    n = len(y_test)
    if n > 0:
        se = np.sqrt(acc * (1 - acc) / n)
        ci_lower = max(0.0, acc - 1.96 * se)
        ci_upper = min(1.0, acc + 1.96 * se)
    else:
        ci_lower, ci_upper = 0.0, 0.0

    return acc, (ci_lower, ci_upper)
