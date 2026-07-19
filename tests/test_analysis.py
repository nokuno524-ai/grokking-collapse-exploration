import pytest
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.phase_transition import find_grokking_step
from src.model import ModularArithmeticTransformer
from analysis.loss_landscape import filter_normalize_direction, get_random_direction

def test_find_grokking_step():
    # Test typical grokking pattern
    history_grok = [
        {"step": 100, "test_acc": 0.1, "weight_norm": 10.0},
        {"step": 200, "test_acc": 0.1, "weight_norm": 12.0},
        {"step": 300, "test_acc": 0.95, "weight_norm": 15.0}, # Groks here
        {"step": 400, "test_acc": 1.0, "weight_norm": 15.5},
    ]

    step, ci = find_grokking_step(history_grok)
    assert step == 300
    assert ci == (200, 400)

    # Test collapse pattern (never groks)
    history_fail = [
        {"step": 100, "test_acc": 0.1, "weight_norm": 10.0},
        {"step": 200, "test_acc": 0.1, "weight_norm": 12.0},
        {"step": 300, "test_acc": 0.15, "weight_norm": 15.0},
    ]

    step_fail, ci_fail = find_grokking_step(history_fail)
    assert step_fail == -1
    assert ci_fail is None

def test_filter_normalization():
    model = ModularArithmeticTransformer(d_model=32, n_heads=2, d_ff=64)
    direction = get_random_direction(model)
    norm_dir = filter_normalize_direction(direction, model)

    # Ensure norms match for each parameter block
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            p_norm = torch.norm(p).item()
            d_norm = torch.norm(norm_dir[idx]).item()

            # Use small tolerance due to float arithmetic
            if d_norm > 0:
                assert np.isclose(p_norm, d_norm, rtol=1e-3, atol=1e-5)
            idx += 1
