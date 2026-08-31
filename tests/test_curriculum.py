import pytest
from src.curriculum.schedules import get_schedule

def test_constant_schedule():
    sched = get_schedule('constant', 0.5, 1.0, 100)
    assert sched(0) == 0.5
    assert sched(50) == 0.5
    assert sched(100) == 0.5

def test_linear_schedule():
    sched = get_schedule('linear', 1.0, 0.0, 100)
    assert sched(0) == 1.0
    assert sched(50) == 0.5
    assert sched(100) == 0.0
    assert sched(150) == 0.0

def test_cosine_schedule():
    sched = get_schedule('cosine', 1.0, 0.0, 100)
    assert sched(0) == 1.0
    assert sched(100) == 0.0
    assert sched(150) == 0.0
    assert 0.49 < sched(50) < 0.51

def test_step_schedule():
    sched = get_schedule('step', 1.0, 0.0, 100)
    assert sched(0) == 1.0
    assert sched(49) == 1.0
    assert sched(50) == 0.0
    assert sched(100) == 0.0

def test_monotonicity():
    linear_sched = get_schedule('linear', 1.0, 0.0, 100)
    for t in range(99):
        assert linear_sched(t) >= linear_sched(t + 1)

    cosine_sched = get_schedule('cosine', 1.0, 0.0, 100)
    for t in range(99):
        assert cosine_sched(t) >= cosine_sched(t + 1)

    linear_sched_up = get_schedule('linear', 0.0, 1.0, 100)
    for t in range(99):
        assert linear_sched_up(t) <= linear_sched_up(t + 1)

    cosine_sched_up = get_schedule('cosine', 0.0, 1.0, 100)
    for t in range(99):
        assert cosine_sched_up(t) <= cosine_sched_up(t + 1)

def test_config_parsing():
    from src.train import TrainConfig
    config = TrainConfig(
        curriculum_schedule='linear',
        curriculum_start_w=0.8,
        curriculum_end_w=0.2
    )
    assert config.curriculum_schedule == 'linear'
    assert config.curriculum_start_w == 0.8
    assert config.curriculum_end_w == 0.2
