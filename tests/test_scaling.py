import pytest
import os
import shutil
from src.train import TrainConfig, train

def test_scaling_train_config(tmp_path):
    """Smoke test to verify that the training loop works with small network and dataset configs."""
    out_dir = tmp_path / "test_results_smoke"

    config = TrainConfig(
        prime=11,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        max_steps=5,
        train_fraction=0.1,
        batch_size=8,
        condition_name="smoke_test",
        output_dir=str(out_dir),
        eval_every=2,
        log_every=2,
        save_every=5
    )

    state = train(config)

    assert state.step == 5
    assert len(state.history) > 0
    assert "train_loss" in state.history[0]

    # Check if results json was created
    assert os.path.exists(os.path.join(out_dir, "smoke_test", "results.json"))
