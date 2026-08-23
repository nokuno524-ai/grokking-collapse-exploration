import pytest
import os
import shutil
from src.train import train, TrainConfig

def test_early_warning_metrics_smoke():
    output_dir = "tests/test_output"

    # 50 steps on a tiny model, CPU
    config = TrainConfig(
        prime=11,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        max_steps=50,
        eval_every=10,
        log_every=10,
        output_dir=output_dir,
        condition_name="smoke_test"
    )

    try:
        state = train(config)

        # Verify metrics were computed and are finite
        assert len(state.history) == 5 # eval_every=10, max_steps=50

        last_entry = state.history[-1]

        metrics_to_check = [
            "weight_norm",
            "embedding_rank",
            "hidden_activation_rank",
            "attention_entropy",
            "train_test_gap_slope",
            "fourier_concentration"
        ]

        for m in metrics_to_check:
            assert m in last_entry
            val = last_entry[m]
            assert isinstance(val, float)
            assert not isinstance(val, bool)
            import math
            assert not math.isnan(val)
            assert not math.isinf(val)

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
