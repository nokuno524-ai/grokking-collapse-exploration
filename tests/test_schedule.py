import pytest
from src.schedule import ScheduleConfig, PhaseConfig, build_schedule
from src.train import train, TrainConfig

def test_fixed_schedule():
    config = ScheduleConfig(schedule_type="fixed", collapse_level=0.1)
    phases = build_schedule(config, max_steps=100)
    assert len(phases) == 1
    assert phases[0].start_step == 0
    assert phases[0].end_step == 100
    assert phases[0].collapse_level == 0.1

def test_switch_schedule():
    config = ScheduleConfig(
        schedule_type="switch",
        collapse_level=0.1,
        switch_step=50,
        switch_to=0.0
    )
    phases = build_schedule(config, max_steps=100)
    assert len(phases) == 2
    assert phases[0].start_step == 0
    assert phases[0].end_step == 50
    assert phases[0].collapse_level == 0.1

    assert phases[1].start_step == 50
    assert phases[1].end_step == 100
    assert phases[1].collapse_level == 0.0

def test_block_interleaved_schedule():
    config = ScheduleConfig(
        schedule_type="block_interleaved",
        collapse_level=0.1,
        switch_to=0.0,
        block_size=20
    )
    phases = build_schedule(config, max_steps=50)
    assert len(phases) == 3
    assert phases[0].end_step == 20
    assert phases[0].collapse_level == 0.1
    assert phases[1].start_step == 20
    assert phases[1].end_step == 40
    assert phases[1].collapse_level == 0.0
    assert phases[2].start_step == 40
    assert phases[2].end_step == 50
    assert phases[2].collapse_level == 0.1

def test_ramp_up_schedule():
    config = ScheduleConfig(
        schedule_type="ramp_up",
        collapse_level=0.0,
        switch_to=0.5,
        ramp_steps=2000
    )
    phases = build_schedule(config, max_steps=100)
    assert len(phases) == 2  # 2000 // 1000 = 2 phases
    assert phases[0].collapse_level == 0.0
    assert phases[1].collapse_level == 0.25

def test_schedule_invalid_type():
    with pytest.raises(ValueError):
        config = ScheduleConfig(schedule_type="unknown")
        build_schedule(config, max_steps=100)

def test_train_loop_with_schedule():
    """Smoke test to ensure the training loop properly processes a schedule switch."""
    config = TrainConfig(
        max_steps=10,
        eval_every=10,
        schedule_type="switch",
        switch_step=5,
        collapse_level=0.2,
        switch_to=0.0
    )
    state = train(config)

    # Check that it reached the end of training
    assert state.step == 10

def test_ramp_up_schedule_fixed():
    config = ScheduleConfig(
        schedule_type="ramp_up",
        collapse_level=0.0,
        switch_to=0.5,
        ramp_steps=2000
    )
    phases = build_schedule(config, max_steps=5000)
    assert len(phases) == 3  # 2 ramp phases, 1 hold phase
    assert phases[0].collapse_level == 0.0
    assert phases[0].start_step == 0
    assert phases[0].end_step == 1000

    assert phases[1].collapse_level == 0.25
    assert phases[1].start_step == 1000
    assert phases[1].end_step == 2000

    assert phases[2].collapse_level == 0.5
    assert phases[2].start_step == 2000
    assert phases[2].end_step == 5000
