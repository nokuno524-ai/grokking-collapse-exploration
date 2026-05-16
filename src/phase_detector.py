import numpy as np
from typing import List, Optional, Dict, Any, Union

class PhaseTransitionDetector:
    """Detects phase transitions in training metrics."""

    @staticmethod
    def detect_transition(metric_series: Union[List[float], np.ndarray], window: int = 50, threshold: float = 2.0) -> Optional[int]:
        """
        Detects a transition by finding where the difference in sliding window means
        normalized by the variance exceeds a threshold.
        """
        if len(metric_series) < 2 * window:
            return None

        series = np.array(metric_series)

        max_score = 0.0
        transition_step = None

        for i in range(window, len(series) - window):
            window_before = series[i-window:i]
            window_after = series[i:i+window]

            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)

            var_before = np.var(window_before)
            var_after = np.var(window_after)

            # Pooled variance
            pooled_var = (var_before + var_after) / 2.0

            # Avoid division by zero
            if pooled_var < 1e-10:
                score = abs(mean_after - mean_before) * 1e5
            else:
                score = abs(mean_after - mean_before) / np.sqrt(pooled_var)

            if score > threshold and score > max_score:
                max_score = score
                transition_step = i

        return transition_step

    @staticmethod
    def detect_grokking_point(train_acc: Union[List[float], np.ndarray],
                              test_acc: Union[List[float], np.ndarray],
                              train_threshold: float = 0.9,
                              test_threshold: float = 0.9,
                              patience: int = 5) -> Optional[int]:
        """
        Detects the step where test accuracy jumps and stays above threshold,
        given train accuracy is already high.
        """
        if len(train_acc) != len(test_acc):
            raise ValueError("train_acc and test_acc must be same length")

        train_acc = np.array(train_acc)
        test_acc = np.array(test_acc)

        for i in range(len(test_acc) - patience):
            if train_acc[i] >= train_threshold and test_acc[i] >= test_threshold:
                # Check if it stays high
                if np.all(test_acc[i:i+patience] >= test_threshold):
                    return i

        return None

    @staticmethod
    def detect_collapse_point(accuracy_series: Union[List[float], np.ndarray],
                              window: int = 10,
                              drop_threshold: float = 0.1) -> Optional[int]:
        """
        Detects the step where accuracy drops significantly below its maximum so far
        and stays low.
        """
        series = np.array(accuracy_series)
        if len(series) < window:
            return None

        max_so_far = np.maximum.accumulate(series)

        for i in range(len(series) - window):
            drop = max_so_far[i] - series[i]
            if drop >= drop_threshold:
                # Check if it stays low
                window_mean = np.mean(series[i:i+window])
                if max_so_far[i] - window_mean >= drop_threshold:
                    return i

        return None

    @staticmethod
    def compute_phase_labels(metrics_history: List[Dict[str, Any]],
                             train_acc_key: str = "train_acc",
                             test_acc_key: str = "test_acc") -> List[str]:
        """
        Computes a list of phase labels for each step in the metrics history.
        Phases: "memorization", "transition", "grokking", "collapsed", "learning".
        """
        train_acc = [m.get(train_acc_key, 0.0) for m in metrics_history]
        test_acc = [m.get(test_acc_key, 0.0) for m in metrics_history]

        grok_point = PhaseTransitionDetector.detect_grokking_point(train_acc, test_acc)
        collapse_point = PhaseTransitionDetector.detect_collapse_point(test_acc)

        # If it collapses before grokking, or if there's a collapse point
        labels = []
        for i in range(len(metrics_history)):
            if collapse_point is not None and i >= collapse_point:
                labels.append("collapsed")
            elif grok_point is not None and i >= grok_point:
                labels.append("grokking")
            elif train_acc[i] > 0.9 and test_acc[i] < 0.5:
                labels.append("memorization")
            elif grok_point is not None and i >= grok_point - 10:
                # 10 steps before grokking is "transition"
                labels.append("transition")
            else:
                labels.append("learning")

        return labels
