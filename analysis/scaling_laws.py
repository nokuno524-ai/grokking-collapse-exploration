import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = a * x^b using logarithmic regression.
    Returns (a, b) where a is coefficient and b is exponent.
    """
    # Filter out non-positive values to avoid log errors
    valid_idx = (x > 0) & (y > 0)
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]

    if len(x_valid) < 2:
        return 0.0, 0.0

    log_x = np.log(x_valid)
    log_y = np.log(y_valid)

    # np.polyfit returns [b, log_a] for degree 1
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    return a, b

def compute_confidence_intervals(x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence intervals on scaling exponents using scipy.stats.
    Returns lower and upper bounds of the exponent (b).
    """
    valid_idx = (x > 0) & (y > 0)
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]

    if len(x_valid) < 3:
        return 0.0, 0.0

    log_x = np.log(x_valid)
    log_y = np.log(y_valid)

    res = stats.linregress(log_x, log_y)
    b = res.slope
    se = res.stderr

    t_val = stats.t.ppf((1 + confidence) / 2., len(x_valid) - 2)
    margin = t_val * se

    return b - margin, b + margin

def fit_scaling_laws(data_sizes: List[float], model_sizes: List[float], grokking_steps: List[float]) -> Dict[str, Any]:
    """
    Compute scaling exponents for grokking step relative to data size and model size.
    """
    data_sizes = np.array(data_sizes)
    model_sizes = np.array(model_sizes)
    grokking_steps = np.array(grokking_steps)

    a_data, b_data = fit_power_law(data_sizes, grokking_steps)
    ci_data = compute_confidence_intervals(data_sizes, grokking_steps)

    a_model, b_model = fit_power_law(model_sizes, grokking_steps)
    ci_model = compute_confidence_intervals(model_sizes, grokking_steps)

    return {
        'data_scaling': {
            'coefficient': a_data,
            'exponent': b_data,
            'ci_lower': ci_data[0],
            'ci_upper': ci_data[1]
        },
        'model_scaling': {
            'coefficient': a_model,
            'exponent': b_model,
            'ci_lower': ci_model[0],
            'ci_upper': ci_model[1]
        }
    }

def extrapolate_grokking_step(target_size: float, scaling_law_params: Dict[str, float]) -> float:
    """
    Predict grokking step for a larger model or dataset based on scaling law parameters.
    """
    a = scaling_law_params.get('coefficient', 0.0)
    b = scaling_law_params.get('exponent', 0.0)

    if target_size <= 0 or a == 0:
        return 0.0

    return a * (target_size ** b)
