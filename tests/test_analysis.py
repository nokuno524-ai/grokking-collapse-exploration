import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.statistics import cohen_d, bootstrap_ci

def test_cohen_d():
    # Test identical distributions
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    d = cohen_d(x, y)
    assert abs(d) < 0.5  # Should be small

    # Test shifted distributions
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(2, 1, 100)
    d = cohen_d(x, y)
    assert d < -1.5  # Should be large negative

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)
    mean, lower, upper = bootstrap_ci(data, n_bootstraps=100)

    assert np.isclose(mean, np.mean(data), atol=0.5)
    assert lower < mean
    assert upper > mean
    # True mean (10) should be within CI most of the time
    assert lower < 10 < upper

import json
import shutil
import tempfile

def test_make_figures_synthetic():
    """Test figure generation data loading with a temporary synthetic structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Override the MULTI_SEED_DIR in the module for this test
        import scripts.make_paper_figures as mfig
        original_dir = mfig.MULTI_SEED_DIR
        mfig.MULTI_SEED_DIR = tmp_path

        try:
            # Create synthetic data
            for seed in [42, 43]:
                for cond in mfig.CONDITIONS:
                    cond_dir = tmp_path / str(seed) / cond
                    cond_dir.mkdir(parents=True)

                    # Create fake history
                    history = []
                    for step in range(0, 100, 10):
                        history.append({
                            "step": step,
                            "train_acc": 0.5 + step * 0.001,
                            "test_acc": 0.1 + step * 0.002,
                            "weight_norm": 20 + step * 0.1,
                            "fourier_concentration": 0.05 + step * 0.001
                        })

                    with open(cond_dir / "results.json", "w") as f:
                        json.dump({"history": history}, f)

            # Test loading
            steps, mean_val, lower, upper = mfig.load_multi_seed_data("pure", "train_acc")

            assert steps is not None
            assert len(steps) == 10
            assert mean_val is not None
            assert len(mean_val) == 10
            assert lower is not None
            assert upper is not None

        finally:
            mfig.MULTI_SEED_DIR = original_dir
