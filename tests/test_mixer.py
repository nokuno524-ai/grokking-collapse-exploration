import pytest
import torch
import numpy as np
from src.curriculum.mixer import generate_curriculum_batch

def test_mixer_proportions():
    clean_in = torch.zeros((100, 2), dtype=torch.long)
    clean_tgt = torch.zeros(100, dtype=torch.long)

    collapse_in = torch.ones((100, 2), dtype=torch.long)
    collapse_tgt = torch.ones(100, dtype=torch.long)

    rng = np.random.RandomState(42)
    batch_size = 1000
    w = 0.3

    batch_in, batch_tgt = generate_curriculum_batch(
        clean_in, clean_tgt, collapse_in, collapse_tgt, batch_size, w, rng
    )

    assert batch_in.shape == (1000, 2)
    assert batch_tgt.shape == (1000,)

    n_collapse = batch_tgt.sum().item()
    # Should be around 300, allow some statistical tolerance
    assert 250 <= n_collapse <= 350

    # Check w=0
    batch_in, batch_tgt = generate_curriculum_batch(
        clean_in, clean_tgt, collapse_in, collapse_tgt, batch_size, 0.0, rng
    )
    assert batch_tgt.sum().item() == 0

    # Check w=1
    batch_in, batch_tgt = generate_curriculum_batch(
        clean_in, clean_tgt, collapse_in, collapse_tgt, batch_size, 1.0, rng
    )
    assert batch_tgt.sum().item() == 1000
