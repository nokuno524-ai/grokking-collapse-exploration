import numpy as np
from src.analysis.stats import detect_grokking_cliff, bootstrap_cliff, wilson_score_interval, compare_conditions

def test_detect_sharp_cliff():
    # Create a sharp step function
    arr = np.concatenate([
        np.full(50, 0.1),
        np.full(50, 0.99)
    ])
    res = detect_grokking_cliff(arr)
    assert res is not None
    cliff_idx, mag = res
    assert cliff_idx == 49
    assert mag > 0.8

def test_detect_noisy_cliff():
    np.random.seed(42)
    # Create a noisy step function
    arr = np.concatenate([
        np.random.normal(0.1, 0.05, 50),
        np.linspace(0.1, 0.9, 10), # transition phase
        np.random.normal(0.9, 0.05, 40)
    ])
    res = detect_grokking_cliff(arr)
    assert res is not None
    cliff_idx, mag = res
    # Transition happens somewhere in the [50, 60] range
    assert 45 <= cliff_idx <= 65
    assert mag > 0.05

def test_no_cliff():
    np.random.seed(42)
    # Never groks (memorizing only)
    arr = np.random.normal(0.1, 0.05, 100)
    res = detect_grokking_cliff(arr)
    assert res is None

    # Starts out perfect and stays there (no jump)
    arr2 = np.random.normal(0.95, 0.01, 100)
    res2 = detect_grokking_cliff(arr2)
    assert res2 is None

def test_ci_coverage():
    np.random.seed(42)
    # 5 runs that grok around step 50
    trajectories = []
    for _ in range(5):
        jump_step = np.random.randint(45, 55)
        traj = np.concatenate([
            np.full(jump_step, 0.1),
            np.full(100 - jump_step, 0.95)
        ])
        trajectories.append(traj)

    res = bootstrap_cliff(trajectories)

    assert res['grok_rate'] == 1.0
    assert res['n_grokked'] == 5
    assert res['mean_step'] is not None
    assert 40 <= res['mean_step'] <= 60
    assert res['ci_step_lower'] is not None
    assert res['ci_step_upper'] is not None
    assert res['ci_step_lower'] <= res['mean_step'] <= res['ci_step_upper']

    assert res['mean_magnitude'] is not None
    assert res['mean_magnitude'] > 0.8
    assert res['ci_mag_lower'] is not None
    assert res['ci_mag_upper'] is not None
    assert res['ci_mag_lower'] <= res['mean_magnitude'] <= res['ci_mag_upper']

def test_compare_conditions():
    np.random.seed(42)
    group_a = []
    for _ in range(5):
        jump_step = np.random.randint(20, 30)
        traj = np.concatenate([np.full(jump_step, 0.1), np.full(100 - jump_step, 0.95)])
        group_a.append(traj)

    group_b = []
    for _ in range(5):
        jump_step = np.random.randint(70, 80)
        traj = np.concatenate([np.full(jump_step, 0.1), np.full(100 - jump_step, 0.95)])
        group_b.append(traj)

    res = compare_conditions(group_a, group_b)

    assert res['grok_rate_a'] == 1.0
    assert res['grok_rate_b'] == 1.0
    assert res['mw_p_value'] is not None
    assert res['mw_p_value'] < 0.05 # Should be significant difference in transition steps
    assert res['step_effect_size'] is not None
    assert abs(res['step_effect_size']) == 1.0 # completely separated
    assert res['final_acc_mw_p'] == 1.0 # final accuracy is the same (both hit 0.95)
