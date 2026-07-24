import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Any, Tuple

def extract_early_features(history: List[Dict[str, Any]], n_steps: int = 10) -> Dict[str, float]:
    """
    Extracts features from early training dynamics to predict grokking.
    Features: train/test loss gap, weight norm trajectory slope, gradient noise scale (if available), attention entropy.
    If the history length is less than n_steps, it uses the available history.
    """
    if not history:
        return {}

    early_history = history[:n_steps]

    train_losses = [entry.get('train_loss', 0.0) for entry in early_history]
    test_losses = [entry.get('test_loss', 0.0) for entry in early_history]
    weight_norms = [entry.get('weight_norm', 0.0) for entry in early_history]
    grad_noise = [entry.get('grad_noise_scale', 0.0) for entry in early_history]
    attn_entropy = [entry.get('attention_entropy', 0.0) for entry in early_history]

    # 1. Train/Test Loss Gap
    final_gap = test_losses[-1] - train_losses[-1] if train_losses and test_losses else 0.0
    mean_gap = np.mean([te - tr for te, tr in zip(test_losses, train_losses)]) if train_losses and test_losses else 0.0

    # 2. Weight norm trajectory slope
    if len(weight_norms) > 1:
        x = np.arange(len(weight_norms))
        slope = np.polyfit(x, weight_norms, 1)[0]
    else:
        slope = 0.0

    # 3. Gradient noise scale (mean over early steps)
    mean_grad_noise = np.mean(grad_noise) if grad_noise else 0.0

    # 4. Attention entropy (mean over early steps)
    mean_attn_entropy = np.mean(attn_entropy) if attn_entropy else 0.0

    return {
        "final_loss_gap": final_gap,
        "mean_loss_gap": mean_gap,
        "weight_norm_slope": slope,
        "mean_grad_noise": mean_grad_noise,
        "mean_attn_entropy": mean_attn_entropy
    }

def train_grokking_predictor(
    histories: List[List[Dict[str, Any]]],
    labels: List[int],
    n_steps: int = 10
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Trains a logistic regression classifier to predict whether grokking will occur.
    Returns the trained model and a dictionary of feature importances.
    """
    if not histories or not labels:
        raise ValueError("Histories and labels must not be empty.")

    features_list = [extract_early_features(h, n_steps) for h in histories]

    # Ensure all features have the same keys
    feature_names = list(features_list[0].keys())

    X = []
    for f in features_list:
        X.append([f[name] for name in feature_names])

    X = np.array(X)
    y = np.array(labels)

    model = LogisticRegression(random_state=42)
    # Check if there's only 1 class
    if len(np.unique(y)) > 1:
        model.fit(X, y)
        importances = model.coef_[0]
    else:
        # Dummy fit if only one class
        model.classes_ = np.unique(y)
        model.coef_ = np.zeros((1, X.shape[1]))
        model.intercept_ = np.zeros((1,))
        importances = model.coef_[0]

    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}

    return model, importance_dict

def predict_grokking(model: LogisticRegression, history: List[Dict[str, Any]], n_steps: int = 10) -> int:
    """
    Predicts whether grokking will occur for a given history.
    """
    features = extract_early_features(history, n_steps)
    # Note: Assumes features keys order matches the training time
    X = np.array([[features[name] for name in features.keys()]])
    return int(model.predict(X)[0])
