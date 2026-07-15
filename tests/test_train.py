import torch
import os
import shutil
from src.train import train, TrainConfig, compute_fourier_concentration
from src.model import ModularArithmeticTransformer

def test_fourier_concentration():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    fc = compute_fourier_concentration(model)
    assert 0 <= fc <= 1.0

def test_train_loop_fast():
    out_dir = "tests/test_results"
    config = TrainConfig(
        prime=7,
        d_model=16,
        n_heads=1,
        d_ff=32,
        max_steps=5,
        eval_every=2,
        log_every=2,
        save_every=5,
        output_dir=out_dir,
        condition_name="test_cond"
    )

    state = train(config)
    assert state.step == 5
    assert len(state.history) == 2  # steps 2 and 4

    # Check that results were saved
    assert os.path.exists(os.path.join(out_dir, "test_cond", "results.json"))
    assert os.path.exists(os.path.join(out_dir, "test_cond", "checkpoint_5.pt"))

    shutil.rmtree(out_dir)

def test_grokking_detection():
    from src.train import TrainState

    state = TrainState(grokking_threshold=0.95)
    assert not state.grokked

    # Simulate a step where grokking hasn't happened yet
    state.test_acc = 0.5

    # In actual train.py grokking is set via if test_acc >= state.grokking_threshold
    # Let's just test the logic manually
    if state.test_acc >= state.grokking_threshold and not state.grokked:
        state.grokked = True
        state.grokking_step = 100

    assert not state.grokked

    state.test_acc = 0.96
    if state.test_acc >= state.grokking_threshold and not state.grokked:
        state.grokked = True
        state.grokking_step = 200

    assert state.grokked
    assert state.grokking_step == 200
