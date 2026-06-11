import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def detect_grokking_phase(test_accuracy: List[float], threshold: float = 0.95, window_size: int = 50) -> Optional[int]:
    """
    Automatic detection of grokking phase transitions using test accuracy curves.
    Grokking is defined as the first training step where test accuracy exceeds a threshold
    and remains above that threshold for a specified window of steps.

    Args:
        test_accuracy: List of test accuracy values over time.
        threshold: The accuracy threshold to define "grokking".
        window_size: Number of subsequent steps that must remain above threshold.

    Returns:
        The index (step) where grokking occurs, or None if it doesn't grok.
    """
    if len(test_accuracy) < window_size:
        return None

    for i in range(len(test_accuracy) - window_size + 1):
        if test_accuracy[i] >= threshold:
            # Check if it remains above threshold for the window
            if all(acc >= threshold for acc in test_accuracy[i:i+window_size]):
                return i

    return None


def detect_collapse_onset(test_accuracy: List[float], baseline_accuracy: List[float], degradation_threshold: float = 0.1, min_steps: int = 10) -> Optional[int]:
    """
    Detect collapse onset - when performance starts degrading significantly compared to a baseline.

    Args:
        test_accuracy: List of test accuracy values for the current model.
        baseline_accuracy: List of test accuracy values for a pure/baseline model.
        degradation_threshold: The accuracy difference that indicates collapse.
        min_steps: Minimum number of consecutive steps the degradation must be observed.

    Returns:
        The index (step) where collapse onset occurs, or None if no collapse detected.
    """
    min_len = min(len(test_accuracy), len(baseline_accuracy))
    if min_len < min_steps:
        return None

    consecutive_degraded = 0
    onset_index = -1

    for i in range(min_len):
        if baseline_accuracy[i] - test_accuracy[i] >= degradation_threshold:
            if consecutive_degraded == 0:
                onset_index = i
            consecutive_degraded += 1
            if consecutive_degraded >= min_steps:
                return onset_index
        else:
            consecutive_degraded = 0
            onset_index = -1

    return None


def compute_critical_points(test_accuracy_dict: Dict[float, List[float]], grokking_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Compute critical points across different severity levels.

    Args:
        test_accuracy_dict: Dictionary mapping severity level -> list of test accuracies.
        grokking_threshold: Accuracy threshold for grokking.

    Returns:
        Dictionary containing:
        - 'collapse_threshold': The maximum severity level before grokking completely fails.
        - 'grokking_onsets': Dictionary mapping severity level -> grokking onset step.
        - 'recovery_potential': Metric indicating how much model can recover (max acc - min acc after collapse).
    """
    grokking_onsets = {}
    collapse_threshold = 0.0

    # Sort severity levels to find the threshold correctly
    severity_levels = sorted(list(test_accuracy_dict.keys()))

    for severity in severity_levels:
        acc_curve = test_accuracy_dict[severity]
        onset = detect_grokking_phase(acc_curve, threshold=grokking_threshold)
        grokking_onsets[severity] = onset

        if onset is not None:
            collapse_threshold = max(collapse_threshold, severity)

    # Calculate recovery potential for the highest severity level that fails to grok
    recovery_potential = 0.0
    failed_severities = [s for s, o in grokking_onsets.items() if o is None]

    if failed_severities:
        # Pick the lowest severity that failed
        first_failed_severity = min(failed_severities)
        acc_curve = test_accuracy_dict[first_failed_severity]
        if len(acc_curve) > 0:
            max_acc = max(acc_curve)
            min_acc = min(acc_curve)
            recovery_potential = max_acc - min_acc

    return {
        'collapse_threshold': collapse_threshold,
        'grokking_onsets': grokking_onsets,
        'recovery_potential': recovery_potential
    }
