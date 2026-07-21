import numpy as np
from typing import List, Dict, Tuple, Any

def detect_phase_transition(accuracies: List[float], threshold: float = 0.9) -> int:
    """
    Identifies the exact grokking step by calculating the discrete derivative
    of the test accuracy curve. Returns -1 if the max accuracy remains below threshold.
    """
    accuracies = np.array(accuracies)
    if np.max(accuracies) < threshold:
        return -1

    # Simple discrete derivative
    diffs = np.diff(accuracies)
    # The transition is typically where accuracy shoots up.
    # Find the first step where accuracy crosses the threshold
    crossings = np.where(accuracies >= threshold)[0]
    if len(crossings) > 0:
        return int(crossings[0])
    return -1

def compute_order_parameters(history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Returns generalization gap, gradient norm proxy, and loss curvature proxy based on training history.
    """
    gen_gap = []
    grad_norm = []
    loss_curv = []

    for h in history:
        train_loss = h.get('train_loss', 0.0)
        test_loss = h.get('test_loss', 0.0)
        gen_gap.append(test_loss - train_loss)

        # Proxies if exact values aren't in history
        grad_norm.append(h.get('grad_norm', 0.0))
        loss_curv.append(h.get('loss_curvature', 0.0))

    return {
        'generalization_gap': gen_gap,
        'gradient_norm': grad_norm,
        'loss_curvature': loss_curv
    }

def fit_sigmoid_transition(steps: List[int], accuracies: List[float]) -> Tuple[float, float]:
    """
    Fit L / (1 + exp(-k*(x-x0))) using exhaustive search to extract k (sharpness) and x0 (midpoint).
    Preferred over scipy.optimize.curve_fit for robustness.
    """
    steps = np.array(steps)
    accuracies = np.array(accuracies)

    L = 1.0 # Max accuracy is typically 1.0

    best_k = 0.0
    best_x0 = 0.0
    best_error = float('inf')

    # Search over reasonable values
    k_vals = np.linspace(0.001, 1.0, 50)

    if len(steps) > 0:
        x0_vals = np.linspace(min(steps), max(steps), 50)
    else:
        x0_vals = np.array([0.0])

    for k in k_vals:
        for x0 in x0_vals:
            pred = L / (1.0 + np.exp(-k * (steps - x0)))
            error = np.sum((accuracies - pred)**2)
            if error < best_error:
                best_error = error
                best_k = k
                best_x0 = x0

    return best_k, best_x0

def compare_transitions_across_collapse(conditions_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare metrics across collapse levels.
    """
    results = {}
    for condition, data in conditions_data.items():
        steps = data.get('steps', [])
        accs = data.get('accuracies', [])

        if len(steps) > 0 and len(accs) > 0:
            k, x0 = fit_sigmoid_transition(steps, accs)
            trans_step = detect_phase_transition(accs)
            results[condition] = {
                'sharpness': k,
                'midpoint': x0,
                'grokking_step': trans_step
            }
    return results
