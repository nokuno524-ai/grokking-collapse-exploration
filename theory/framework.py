import numpy as np

def compute_mutual_information(collapse_severity: float, snr: float) -> float:
    """
    Computes theoretical mutual information between weights and the true task.
    In the context of model collapse acting as noise injection, higher collapse
    severity reduces mutual information.

    Args:
        collapse_severity: Float in [0, 1] indicating ratio of synthetic/noisy data.
        snr: Signal-to-noise ratio of the underlying data distribution.

    Returns:
        Estimated mutual information (in nats or bits, conceptually).
    """
    # Simple theoretical model: MI scales with log(1 + SNR_eff)
    # Effective SNR degrades with collapse_severity.
    effective_snr = snr * (1.0 - collapse_severity)
    return 0.5 * np.log2(1 + effective_snr)

def weight_norm_trajectory(t: np.ndarray, wd: float, eta: float, initial_norm: float) -> np.ndarray:
    """
    Dynamical systems model of weight norm trajectory as a differential equation solution.
    Models the competition between weight decay and noise gradients.

    Args:
        t: Array of time steps.
        wd: Weight decay coefficient (lambda).
        eta: Label corruption rate (collapse rate).
        initial_norm: Initial weight norm at t=0.

    Returns:
        Array of expected weight norms at each time step.
    """
    # Simplified ODE solution: dw/dt = -wd * w + eta
    # Represents drift towards 0 from decay, and drift outward from noise.
    # Steady state is eta / wd.
    steady_state = eta / (wd + 1e-9)
    # w(t) = w_ss + (w_0 - w_ss) * exp(-wd * t)
    return steady_state + (initial_norm - steady_state) * np.exp(-wd * t)


class PhaseTransitionModel:
    """
    Models the phase transition of grokking.
    Critical point is determined by a collapse threshold.
    """
    def __init__(self, critical_threshold: float = 0.1):
        self.critical_threshold = critical_threshold

    def is_grokking_expected(self, collapse_severity: float) -> bool:
        """
        Returns True if grokking is expected (collapse is below critical threshold).
        """
        return collapse_severity < self.critical_threshold


def predict_grokking(collapse_severity: float, wd: float) -> bool:
    """
    Predicts if the model will grok given the collapse severity and weight decay.
    Based on the empirical grid from Experiment C, where a sharp cliff exists
    at a critical noise level, slightly modulated by wd.
    """
    # Simple predictive model: base threshold around 0.1, lowered by extreme wd
    base_threshold = 0.12
    # Adjust threshold slightly by wd (if wd is too high, it might prevent grokking even at 0 noise,
    # but for typical values like 0.3-1.0, threshold shifts slightly)
    if wd > 2.0:
        return False

    threshold = base_threshold - (wd - 1.0) * 0.02
    return collapse_severity < threshold
