import numpy as np
from typing import List, Dict, Tuple, Any

def compute_mutual_information(representations: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if len(representations.shape) > 1:
        mi_total = 0.0
        d = representations.shape[1]
        for i in range(d):
            mi_total += _compute_mi_1d(representations[:, i], labels, bins)
        return mi_total / d
    else:
        return _compute_mi_1d(representations, labels, bins)

def _compute_mi_1d(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    hist_2d, _, _ = np.histogram2d(x, y, bins=[bins, len(np.unique(y))])
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi

def measure_effective_information(representations: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """
    Measure effective information in hidden layers.
    Effective information content in label distributions is calculated in bits as
    max entropy minus current Shannon entropy. We apply this to the joint distribution
    or just use the MI as a proxy.
    Here we calculate the max entropy of the representation - its Shannon entropy.
    """
    if len(representations.shape) > 1:
        eff_total = 0.0
        d = representations.shape[1]
        for i in range(d):
            hist, _ = np.histogram(representations[:, i], bins=bins)
            p = hist / float(np.sum(hist))
            p = p[p > 0]
            shannon_entropy = -np.sum(p * np.log2(p))
            max_entropy = np.log2(bins)
            eff_total += (max_entropy - shannon_entropy)
        return eff_total / d
    else:
        hist, _ = np.histogram(representations, bins=bins)
        p = hist / float(np.sum(hist))
        p = p[p > 0]
        shannon_entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(bins)
        return max_entropy - shannon_entropy

def compute_compression_tradeoff(history: List[Dict[str, Any]], model_states: List[np.ndarray], labels: np.ndarray, bins: int = 10) -> Dict[str, List[float]]:
    """
    Analyze MI over time vs accuracy.
    """
    steps = [h.get('step', i) for i, h in enumerate(history)]
    accs = [h.get('test_acc', 0.0) for h in history]

    mis = []
    for state in model_states:
        if state is not None and len(state) > 0:
            mis.append(compute_mutual_information(state, labels, bins))
        else:
            mis.append(0.0)

    return {
        'steps': steps,
        'accuracy': accs,
        'mutual_information': mis
    }
