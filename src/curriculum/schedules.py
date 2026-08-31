import math
from typing import Callable

def get_schedule(schedule_type: str, start_w: float, end_w: float, max_steps: int) -> Callable[[int], float]:
    """
    Returns a function w(t) that yields the mixture weight at step t.
    w(t) is the proportion of data that comes from the 'collapsed' dataset.

    Args:
        schedule_type: 'constant', 'linear', 'cosine', or 'step'
        start_w: initial mixture weight (at step 0)
        end_w: final mixture weight (at step max_steps)
        max_steps: total training steps
    """
    if schedule_type == 'constant':
        return lambda t: start_w
    elif schedule_type == 'linear':
        def linear(t: int) -> float:
            if t >= max_steps:
                return end_w
            progress = t / max_steps
            return start_w + (end_w - start_w) * progress
        return linear
    elif schedule_type == 'cosine':
        def cosine(t: int) -> float:
            if t >= max_steps:
                return end_w
            progress = t / max_steps
            # Cosine from 0 to pi gives 1 to -1. We map to 0 to 1
            cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
            # Start at start_w (when cosine_val is 1) to end_w (when cosine_val is 0)
            return end_w + (start_w - end_w) * cosine_val
        return cosine
    elif schedule_type == 'step':
        # Change weight at 50% of training
        def step(t: int) -> float:
            if t >= max_steps / 2:
                return end_w
            return start_w
        return step
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
