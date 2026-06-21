import pytest
import torch
import os
import shutil
from src.train import train, TrainConfig

def test_training_loop_loss_decreases():
    # Setup output dir
    output_dir = "tests/test_results"
    os.makedirs(output_dir, exist_ok=True)

    config = TrainConfig(
        prime=11,  # Small prime for fast test
        d_model=32,
        n_heads=2,
        d_ff=64,
        max_steps=20,
        eval_every=5,
        save_every=20,
        batch_size=32,
        condition_name="test_condition",
        output_dir=output_dir,
        seed=42,
    )

    state = train(config)

    assert state.step == 20
    assert len(state.history) == 4  # eval at 5, 10, 15, 20

    # Check that training loss decreased
    first_loss = state.history[0]["train_loss"]
    last_loss = state.history[-1]["train_loss"]

    assert last_loss < first_loss

    # Cleanup
    shutil.rmtree(output_dir)
