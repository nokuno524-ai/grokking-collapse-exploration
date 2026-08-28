import numpy as np
from typing import List, Dict, Any

def compute_rolling_features(history: List[Dict[str, Any]], window_size: int = 5) -> Dict[str, List[float]]:
    """
    Computes early-warning predictors over a rolling window.

    Predictors extracted:
    - loss_gap: test_loss - train_loss
    - weight_norm_slope: rolling log-linear slope of weight_norm
    - effective_rank: directly from embedding_rank (or other rank metrics if present)
    - test_acc_curvature: rolling second derivative estimate of test_acc
    - activation_sparsity: from activation_sparsity if present, else NaN
    - gradient_norm: from gradient_norm if present, else NaN

    Output length is equal to history length. First `window_size-1` elements
    for rolling features like slope and curvature will be NaN.
    """
    if not history:
        return {}

    n = len(history)
    features = {
        "step": [h.get("step", 0) for h in history],
        "loss_gap": np.full(n, np.nan),
        "weight_norm_slope": np.full(n, np.nan),
        "effective_rank": np.full(n, np.nan),
        "test_acc_curvature": np.full(n, np.nan),
        "activation_sparsity": np.full(n, np.nan),
        "gradient_norm": np.full(n, np.nan)
    }

    test_loss = np.array([h.get("test_loss", np.nan) for h in history])
    train_loss = np.array([h.get("train_loss", np.nan) for h in history])
    features["loss_gap"] = test_loss - train_loss

    weight_norm = np.array([h.get("weight_norm", np.nan) for h in history])
    rank = np.array([h.get("embedding_rank", np.nan) for h in history])
    test_acc = np.array([h.get("test_acc", np.nan) for h in history])
    act_spars = np.array([h.get("activation_sparsity", np.nan) for h in history])
    grad_norm = np.array([h.get("gradient_norm", np.nan) for h in history])

    features["effective_rank"] = rank
    features["activation_sparsity"] = act_spars
    features["gradient_norm"] = grad_norm

    for i in range(window_size - 1, n):
        # Weight norm slope (log-linear fit)
        y_wn = weight_norm[i - window_size + 1 : i + 1]
        if not np.any(np.isnan(y_wn)) and np.all(y_wn > 0):
            x = np.arange(window_size)
            slope, _ = np.polyfit(x, np.log(y_wn), 1)
            features["weight_norm_slope"][i] = slope

    # Curvature requires at least 3 points, we compute it over the window
    for i in range(max(2, window_size - 1), n):
        y_acc = test_acc[i - window_size + 1 : i + 1]
        if not np.any(np.isnan(y_acc)) and len(y_acc) >= 3:
            # We estimate curvature as the coefficient of x^2 in a quadratic fit
            x = np.arange(len(y_acc))
            coeffs = np.polyfit(x, y_acc, 2)
            features["test_acc_curvature"][i] = coeffs[0]  # Coefficient of x^2

    return features
