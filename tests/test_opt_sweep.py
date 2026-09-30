import pytest
import os
import sys
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from optimizer_sweep.opt_stats import compute_interaction_stats
from optimizer_sweep.opt_boundary_map import generate_boundary_map

def test_compute_interaction_stats():
    """Test ANOVA decomposition with synthetic data."""
    # Create a synthetic dataset with known effects
    results = []

    # Base setup: grand mean ~20000
    # Effects: AdamW (-5000), SGD (+5000)
    # Effects: WD 0.1 (-2000), WD 1.0 (0), WD 3.0 (+2000)
    # Effects: pure (-3000), collapsed (+3000)

    for opt in ['adamw', 'sgd']:
        opt_effect = -5000 if opt == 'adamw' else 5000

        for wd in [0.1, 1.0, 3.0]:
            wd_effect = -2000 if wd == 0.1 else (0 if wd == 1.0 else 2000)

            for cond in ['pure', 'medium_collapse']:
                cond_effect = -3000 if cond == 'pure' else 3000

                # Create 3 seeds per cell to allow variance and testing
                for seed in range(3):
                    # Base step + effects + noise
                    noise = (seed - 1) * 500 # -500, 0, 500
                    step = 20000 + opt_effect + wd_effect + cond_effect + noise

                    results.append({
                        'optimizer': opt,
                        'wd': wd,
                        'lr': 1e-3,
                        'condition': cond,
                        'seed': seed,
                        'grokked': True,
                        'grokking_step': step,
                        'final_test_acc': 1.0
                    })

    stats = compute_interaction_stats(results)

    # Grand mean should be exactly 20000
    assert abs(stats['grand_mean'] - 20000) < 1e-5

    # Check main effects
    assert abs(stats['main_effects']['opt=adamw'] - (-5000)) < 1e-5
    assert abs(stats['main_effects']['opt=sgd'] - 5000) < 1e-5

    assert abs(stats['main_effects']['wd=0.1'] - (-2000)) < 1e-5
    assert abs(stats['main_effects']['wd=1.0'] - 0) < 1e-5
    assert abs(stats['main_effects']['wd=3.0'] - 2000) < 1e-5

    assert abs(stats['main_effects']['cond=pure'] - (-3000)) < 1e-5
    assert abs(stats['main_effects']['cond=medium_collapse'] - 3000) < 1e-5

    # Check effect sizes (they should all be > 0)
    assert stats['effect_sizes']['eta2_optimizer'] > 0
    assert stats['effect_sizes']['eta2_wd'] > 0
    assert stats['effect_sizes']['eta2_condition'] > 0

    # Check multiple comparisons
    # Pure vs collapsed should be significant for both optimizers
    assert stats['multiple_comparisons']['opt=adamw: pure vs collapsed']['significant']
    assert stats['multiple_comparisons']['opt=sgd: pure vs collapsed']['significant']

def test_boundary_map_generation():
    """Test boundary map generation handles missing/NaN correctly."""
    results = [
        # Successful run
        {'optimizer': 'adamw', 'wd': 1.0, 'lr': 1e-3, 'condition': 'pure', 'seed': 0,
         'grokked': True, 'grokking_step': 10000, 'final_test_acc': 1.0},
        # Unsuccessful run
        {'optimizer': 'adamw', 'wd': 1.0, 'lr': 1e-3, 'condition': 'pure', 'seed': 1,
         'grokked': False, 'grokking_step': None, 'final_test_acc': 0.4},
        # Successful run
        {'optimizer': 'adamw', 'wd': 1.0, 'lr': 1e-3, 'condition': 'pure', 'seed': 2,
         'grokked': True, 'grokking_step': 12000, 'final_test_acc': 1.0},
    ]

    df = generate_boundary_map(results)

    # We expect 1 row for this combination
    assert len(df) == 1

    row = df.iloc[0]

    # Success rate should be 2/3
    assert abs(row['grokking_success_rate'] - (2/3)) < 1e-5

    # Mean step should be (10000 + 12000) / 2 = 11000 (ignoring the failure)
    assert abs(row['mean_grokking_step'] - 11000) < 1e-5

    # Bootstrap CI should be calculated since we have 2 successful seeds
    assert not np.isnan(row['ci_low'])
    assert not np.isnan(row['ci_high'])

def test_degenerate_grid():
    """Test edge case: missing cells, single condition."""
    results = [
        # Only one condition, only one seed
        {'optimizer': 'adamw', 'wd': 1.0, 'lr': 1e-3, 'condition': 'pure', 'seed': 0,
         'grokked': True, 'grokking_step': 10000, 'final_test_acc': 1.0},
    ]

    # Stats should handle this gracefully
    stats = compute_interaction_stats(results)

    assert stats['grand_mean'] == 10000

    # No comparisons possible with only 1 condition/cell
    assert 'multiple_comparisons' not in stats or not stats['multiple_comparisons']

    # Boundary map should handle this gracefully
    df = generate_boundary_map(results)

    assert len(df) == 1
    # Only 1 seed, so CI should be NaN
    assert np.isnan(df.iloc[0]['ci_low'])
