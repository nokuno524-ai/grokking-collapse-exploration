import pytest
import numpy as np
from src.data import DatasetConfig, generate_modular_arithmetic, apply_collapse

def test_regeneration_tree_logic():
    # Small test config
    prime = 5
    cfg_base = DatasetConfig(prime=prime, train_fraction=1.0, collapse_level=1.0, collapse_severity=0.9, seed=42)

    # 0 generations (effectively identity)
    cfg_0 = DatasetConfig(prime=prime, train_fraction=1.0, collapse_level=1.0, collapse_severity=0.9, tree_depth=0, seed=42)
    in0, tgt0, _, _ = generate_modular_arithmetic(cfg_0)

    # 1 generation
    cfg_1 = DatasetConfig(prime=prime, train_fraction=1.0, collapse_level=1.0, collapse_severity=0.9, tree_depth=1, seed=42)
    in1, tgt1, _, _ = generate_modular_arithmetic(cfg_1)

    # 3 generations
    cfg_3 = DatasetConfig(prime=prime, train_fraction=1.0, collapse_level=1.0, collapse_severity=0.9, tree_depth=3, seed=42)
    in3, tgt3, _, _ = generate_modular_arithmetic(cfg_3)

    # tree_depth=0 should leave targets uncollapsed
    # The targets should just be (a+b)%p which is uniformly distributed.
    assert len(set(tgt0.tolist())) == prime

    # After multiple generations of severe collapse, the entropy should drop
    def entropy(t):
        from collections import Counter
        c = Counter(t.tolist())
        probs = [v/len(t) for v in c.values()]
        return -sum(p * np.log(p) for p in probs)

    e0 = entropy(tgt0)
    e1 = entropy(tgt1)
    e3 = entropy(tgt3)

    # Severe collapse shrinks the distribution, more depth shrinks it more (or stabilizes at low entropy)
    assert e1 < e0
    assert e3 <= e1

def test_determinism():
    cfg1 = DatasetConfig(prime=7, train_fraction=1.0, collapse_level=0.5, collapse_severity=0.5, tree_depth=2, seed=123)
    cfg2 = DatasetConfig(prime=7, train_fraction=1.0, collapse_level=0.5, collapse_severity=0.5, tree_depth=2, seed=123)

    _, tgt1, _, _ = generate_modular_arithmetic(cfg1)
    _, tgt2, _, _ = generate_modular_arithmetic(cfg2)

    assert (tgt1 == tgt2).all()
