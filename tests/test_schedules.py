import pytest
from src.curriculum.schedules import LinearDecaySchedule, StepPhaseOutSchedule

def test_linear_decay_schedule():
    # Decay from 1.0 to 0.0 over the first 50% of steps
    sched = LinearDecaySchedule(start_fresh=1.0, end_fresh=0.0, end_step_ratio=0.5)

    max_steps = 100
    assert sched.get_fresh_fraction(0, max_steps) == 1.0
    assert sched.get_fresh_fraction(25, max_steps) == 0.5
    assert sched.get_fresh_fraction(50, max_steps) == 0.0
    assert sched.get_fresh_fraction(75, max_steps) == 0.0
    assert sched.get_fresh_fraction(100, max_steps) == 0.0

def test_step_phase_out_schedule():
    sched = StepPhaseOutSchedule(switch_step=50, before_fresh=0.0, after_fresh=1.0)

    max_steps = 100
    assert sched.get_fresh_fraction(0, max_steps) == 0.0
    assert sched.get_fresh_fraction(49, max_steps) == 0.0
    assert sched.get_fresh_fraction(50, max_steps) == 1.0
    assert sched.get_fresh_fraction(99, max_steps) == 1.0

def test_edge_cases():
    sched = LinearDecaySchedule()
    assert sched.get_fresh_fraction(5, max_steps=0) == 0.0
    assert sched.get_fresh_fraction(5, max_steps=1) == 0.0
