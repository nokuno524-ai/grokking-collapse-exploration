import pytest
import torch
from src.curriculum.mixer import DataMixer
from src.curriculum.schedules import ConstantSchedule

def test_data_mixer_exact_composition():
    # Simple deterministic datasets
    fresh_in = torch.arange(10).unsqueeze(1).float()
    fresh_tgt = torch.arange(10).long()

    col_in = torch.arange(10, 20).unsqueeze(1).float()
    col_tgt = torch.arange(10, 20).long()

    # 50% mix
    schedule = ConstantSchedule(0.5)
    batch_size = 4

    mixer = DataMixer(
        fresh_inputs=fresh_in, fresh_targets=fresh_tgt,
        collapsed_inputs=col_in, collapsed_targets=col_tgt,
        schedule=schedule, batch_size=batch_size, seed=42
    )

    batch_in, batch_tgt = mixer.get_batch(step=1, max_steps=100)

    assert batch_in.shape == (4, 1)
    assert batch_tgt.shape == (4,)

    # Values < 10 are fresh, >= 10 are collapsed
    n_fresh = (batch_tgt < 10).sum().item()
    n_col = (batch_tgt >= 10).sum().item()

    # 50% of 4 is exactly 2
    assert n_fresh == 2
    assert n_col == 2

def test_data_mixer_boundaries():
    fresh_in = torch.arange(10).unsqueeze(1).float()
    fresh_tgt = torch.arange(10).long()
    col_in = torch.arange(10, 20).unsqueeze(1).float()
    col_tgt = torch.arange(10, 20).long()

    # All fresh
    schedule_all = ConstantSchedule(1.0)
    mixer_all = DataMixer(fresh_in, fresh_tgt, col_in, col_tgt, schedule_all, 6, seed=42)
    b_in, b_tgt = mixer_all.get_batch(1, 100)
    assert (b_tgt < 10).all().item()

    # All collapsed
    schedule_none = ConstantSchedule(0.0)
    mixer_none = DataMixer(fresh_in, fresh_tgt, col_in, col_tgt, schedule_none, 6, seed=42)
    b_in, b_tgt = mixer_none.get_batch(1, 100)
    assert (b_tgt >= 10).all().item()
