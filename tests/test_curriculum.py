import pytest
import torch
from src.train import TrainConfig, train

def test_curriculum_setup():
    # Just run a few steps to ensure the loop works without crashing
    config = TrainConfig(
        prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1,
        max_steps=5, eval_every=5, log_every=5, save_every=500,
        curriculum_mode="clean_to_collapse", switch_step=2
    )

    state = train(config)
    assert state.step == 5

def test_interleaved_setup():
    config = TrainConfig(
        prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1,
        max_steps=5, eval_every=5, log_every=5, save_every=500,
        curriculum_mode="interleaved", interleaved_ratio=0.5
    )
    state = train(config)
    assert state.step == 5
