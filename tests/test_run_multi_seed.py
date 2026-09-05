import pytest
from src.run_multi_seed_stats import build_tasks, CONDITION_ORDER

def test_build_tasks():
    seeds = [42, 43]
    tasks = build_tasks(seeds)

    # 2 seeds * 5 conditions = 10 tasks
    assert len(tasks) == 10

    # Check that we cycle through conditions correctly for the first seed
    for i, cond in enumerate(CONDITION_ORDER):
        seed, c_name, c_cfg = tasks[i]
        assert seed == 42
        assert c_name == cond

    # Check the second seed
    for i, cond in enumerate(CONDITION_ORDER):
        seed, c_name, c_cfg = tasks[5 + i]
        assert seed == 43
        assert c_name == cond
