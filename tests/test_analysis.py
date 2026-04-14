import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import progress_measures

def test_classify_run():
    # grokking
    hist = [{"test_acc": 0.96}]
    assert progress_measures.classify_run(hist) == "grokking"

    # memorization
    hist = [{"train_acc": 0.99, "test_acc": 0.1}]
    assert progress_measures.classify_run(hist) == "memorization"

    # collapse
    hist = [{"mode_collapse": 0.9}]
    assert progress_measures.classify_run(hist) == "collapse"

    # normal
    hist = [{"train_acc": 0.5, "test_acc": 0.5}]
    assert progress_measures.classify_run(hist) == "normal"
