import math
import numpy as np
from typing import Dict, Tuple, List, Callable, Optional, Any

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit log y = log C + b * log x via OLS.
    Returns (C, b, R^2).
    """
    mask = (y > 0) & (x > 0) & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    A = np.column_stack([np.ones_like(log_x), log_x])
    sol, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    log_c, b = sol
    yhat = A @ sol
    ss_res = ((log_y - yhat) ** 2).sum()
    ss_tot = ((log_y - log_y.mean()) ** 2).sum() if log_y.size > 1 else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(log_c)), float(b), float(r2)

def fit_log_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit log y = a + b * x via OLS.
    Returns (a, b, R^2). Note: a is the intercept (log C), b is the slope.
    """
    mask = (y > 0) & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    x_valid = x[mask]
    log_y = np.log(y[mask])
    A = np.column_stack([np.ones_like(x_valid), x_valid])
    sol, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    a, b = sol
    yhat = A @ sol
    ss_res = ((log_y - yhat) ** 2).sum()
    ss_tot = ((log_y - log_y.mean()) ** 2).sum() if log_y.size > 1 else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)

def bootstrap_ci(
    per_seed: Dict[Tuple[float, int], float],
    fit_func: Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
    param_idx: int = 1
) -> Tuple[float, float]:
    """
    Bootstrap CI on a parameter of the fit function across seeds.
    `param_idx` defaults to 1 (the exponent or slope).
    """
    rng = np.random.RandomState(rng_seed)
    # Extract unique seeds and x_vals (e.g., wds or widths)
    keys = list(per_seed.keys())
    if not keys:
        return float("nan"), float("nan")

    seeds = sorted(list(set([s for _, s in keys])))
    x_vals = sorted(list(set([x for x, _ in keys])))

    bs: List[float] = []
    for _ in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        med_y = []
        valid_x = []
        for x in x_vals:
            ys = [per_seed[(x, s)] for s in sampled_seeds if (x, s) in per_seed]
            ys = [y for y in ys if y > 0]
            if ys:
                med_y.append(float(np.median(ys)))
                valid_x.append(x)

        if len(valid_x) < 2:
            continue

        med_y_arr = np.array(med_y)
        x_arr = np.array(valid_x)

        res = fit_func(x_arr, med_y_arr)
        param = res[param_idx]
        if math.isfinite(param):
            bs.append(param)

    if not bs:
        return float("nan"), float("nan")

    bs.sort()
    lo = bs[int((alpha / 2) * len(bs))]
    hi = bs[int((1 - alpha / 2) * len(bs))]
    return float(lo), float(hi)

def fit_logistic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a simple logistic regression: logit(p) = a + b * x
    Where y is boolean/binary (0 or 1).
    Returns (a, b, pseudo-R^2).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    x_valid = x[mask]
    y_valid = y[mask].astype(float)

    unique_y = np.unique(y_valid)
    if len(unique_y) < 2:
        # Perfect prediction (all 0s or all 1s), cannot fit slope
        return float("nan"), float("nan"), 1.0

    from scipy.optimize import minimize

    def nll(params):
        a, b = params
        logits = a + b * x_valid
        # Avoid overflow
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        # negative log likelihood
        return -np.sum(y_valid * np.log(probs) + (1 - y_valid) * np.log(1 - probs))

    res = minimize(nll, x0=[0.0, 0.0], method='L-BFGS-B', bounds=[(-100, 100), (-100, 100)])
    if not res.success:
        return float("nan"), float("nan"), float("nan")

    a, b = res.x

    # Pseudo R2 (McFadden)
    null_prob = np.mean(y_valid)
    if null_prob <= 0 or null_prob >= 1:
        r2 = 1.0
    else:
        null_ll = np.sum(y_valid * np.log(null_prob) + (1 - y_valid) * np.log(1 - null_prob))
        if null_ll == 0:
            r2 = float("nan")
        else:
            r2 = 1 - (-res.fun / null_ll)

    return float(a), float(b), float(r2)

def bootstrap_logistic_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
    param_idx: int = 1
) -> Tuple[float, float]:
    """
    Bootstrap CI on a parameter of the logistic fit function across random resampling.
    `param_idx` defaults to 1 (the slope).
    """
    rng = np.random.RandomState(rng_seed)
    n_samples = len(x)
    if n_samples < 2:
        return float("nan"), float("nan")

    bs: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sampled_x = x[idx]
        sampled_y = y[idx]

        # Ensure we have both classes, else we can't fit
        if len(np.unique(sampled_y)) < 2:
            continue

        res = fit_logistic(sampled_x, sampled_y)
        param = res[param_idx]
        if math.isfinite(param):
            bs.append(param)

    if not bs:
        return float("nan"), float("nan")

    bs.sort()
    lo = bs[int((alpha / 2) * len(bs))]
    hi = bs[int((1 - alpha / 2) * len(bs))]
    return float(lo), float(hi)
