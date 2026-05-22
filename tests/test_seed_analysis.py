import pytest
import numpy as np
from pathlib import Path

def test_seed_mock_results():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.seed_analysis import mock_seed_results, SEEDS

    stats = mock_seed_results()

    assert "pure" in stats
    assert "severe_collapse" in stats

    assert len(stats["pure"]["grok_step"]) == len(SEEDS)
    assert np.all(np.isnan(stats["severe_collapse"]["grok_step"]))
    assert np.nanmean(stats["pure"]["grok_step"]) > 0
