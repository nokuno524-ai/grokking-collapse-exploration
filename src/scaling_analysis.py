import numpy as np
import scipy.optimize as opt
from typing import List, Dict, Tuple
import json
import os

def run_grokking_sweep(model_sizes: List[int], dataset_qualities: List[float], dummy_mode: bool = False) -> Dict[str, dict]:
    """
    Orchestrates experiments across model sizes and data quality levels.
    In dummy_mode (for testing), generates synthetic mock results.
    Otherwise, this would normally spin off training jobs or load real results.

    Returns a dictionary structured as:
    results[(size, quality)] = {"grokking_step": step, "max_accuracy": acc}
    """
    results = {}
    for size in model_sizes:
        for quality in dataset_qualities:
            if dummy_mode:
                # Mock simulation:
                # Larger models grok faster (fewer steps).
                # Better data quality (higher value means cleaner, or lower means less collapsed)
                # Let's say quality is collapse severity, so lower is better.
                # Threshold behavior: if severity > some critical point, it never groks.

                # Critical severity increases slightly with model size (larger models are more robust)
                critical_quality = 0.2 + (size / 1e6) * 0.1

                if quality > critical_quality:
                    # Fails to grok
                    step = -1
                    acc = 0.6
                else:
                    # Groks
                    # Step count is inversely proportional to model size and increases with severity
                    base_step = 1000
                    step_multiplier = (1.0 / (size / 1e5)) * (1.0 + quality * 10)
                    step = int(base_step * step_multiplier)
                    acc = 0.99

                results[f"{size}_{quality}"] = {
                    "model_size": size,
                    "collapse_severity": quality,
                    "grokking_step": step,
                    "max_accuracy": acc
                }
            else:
                raise NotImplementedError("Real training orchestration not implemented. Use dummy_mode=True.")

    return results

def compute_grokking_threshold(results: Dict[str, dict]) -> Dict[int, float]:
    """
    Finds the critical collapse level (threshold) for each model size.
    Returns a dictionary mapping model_size -> critical_collapse_severity.
    """
    # Group by model size
    size_to_qualities = {}
    for key, data in results.items():
        size = data["model_size"]
        quality = data["collapse_severity"]
        grokked = data["grokking_step"] != -1

        if size not in size_to_qualities:
            size_to_qualities[size] = []
        size_to_qualities[size].append((quality, grokked))

    thresholds = {}
    for size, qualities in size_to_qualities.items():
        # Sort by quality (severity)
        qualities.sort(key=lambda x: x[0])

        # Find transition point
        threshold = None
        for i in range(len(qualities) - 1):
            if qualities[i][1] and not qualities[i+1][1]:
                # Midpoint between the last successful grok and first failure
                threshold = (qualities[i][0] + qualities[i+1][0]) / 2.0
                break

        # If it always groks, threshold is the max tested
        if threshold is None and all(q[1] for q in qualities):
            threshold = qualities[-1][0]
        # If it never groks, threshold is 0
        elif threshold is None:
            threshold = 0.0

        thresholds[size] = threshold

    return thresholds

def power_law(x, a, b):
    return a * (x ** b)

def fit_scaling_law(results: Dict[str, dict]) -> Dict[str, float]:
    """
    Fits power-law relationships between model size and the grokking threshold.
    Model: threshold = a * (size ** b)
    Returns fitted parameters 'a' and 'b'.
    """
    thresholds = compute_grokking_threshold(results)

    sizes = np.array(list(thresholds.keys()), dtype=float)
    thresh_vals = np.array(list(thresholds.values()), dtype=float)

    # Filter out zeros for log-log fit
    valid = thresh_vals > 0
    if sum(valid) < 2:
        return {"a": 0.0, "b": 0.0}

    sizes_valid = sizes[valid]
    thresh_valid = thresh_vals[valid]

    # Linear fit in log-log space: log(thresh) = log(a) + b * log(size)
    log_sizes = np.log(sizes_valid)
    log_thresh = np.log(thresh_valid)

    b, log_a = np.polyfit(log_sizes, log_thresh, 1)
    a = np.exp(log_a)

    return {"a": float(a), "b": float(b)}

if __name__ == "__main__":
    # Test with dummy data
    sizes = [100000, 200000, 500000, 1000000]
    qualities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = run_grokking_sweep(sizes, qualities, dummy_mode=True)
    thresholds = compute_grokking_threshold(results)
    law = fit_scaling_law(results)

    print("Thresholds:", thresholds)
    print("Scaling Law:", law)
