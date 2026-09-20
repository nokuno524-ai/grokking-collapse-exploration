with open("src/data.py", "r") as f:
    content = f.read()

content = content.replace("seed: int = 42\n", "seed: int = 42\n    tree_depth: int = 1  # Number of recursive generation steps\n")

content = content.replace(
"""    # Apply collapse: replace some training examples with "collapsed" outputs
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, p,
            config.collapse_level, config.collapse_severity, rng
        )""",
"""    # Apply collapse: replace some training examples with "collapsed" outputs
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, p,
            config.collapse_level, config.collapse_severity, config.seed, config.tree_depth
        )""")

content = content.replace(
"""def apply_collapse(
    pairs: list, targets: list, prime: int,
    collapse_level: float, collapse_severity: float, rng: np.random.RandomState
) -> Tuple[list, list]:
    \"\"\"
    Simulate model collapse by replacing some targets with outputs from a "collapsed" model.

    A collapsed model has:
    - Narrowed output distribution (favors common results)
    - Occasional errors (assigns probability mass incorrectly)
    - Loss of rare outputs
    \"\"\"
    n_replace = int(len(targets) * collapse_level)
    replace_idx = rng.choice(len(targets), n_replace, replace=False)

    # Compute target frequency distribution
    from collections import Counter
    target_counts = Counter(targets)
    total = len(targets)
    freq = {t: c / total for t, c in target_counts.items()}

    # Create collapsed distribution: amplify common targets, suppress rare ones
    # Use temperature to control severity
    temp = max(0.1, 1.0 - collapse_severity)
    collapsed_probs = {}
    for t in range(prime):
        base_prob = freq.get(t, 1.0 / prime)
        collapsed_probs[t] = base_prob ** (1.0 / temp)

    # Normalize
    total_prob = sum(collapsed_probs.values())
    collapsed_probs = {t: p / total_prob for t, p in collapsed_probs.items()}

    # Sample from collapsed distribution
    collapsed_targets = list(collapsed_probs.keys())
    collapsed_weights = [collapsed_probs[t] for t in collapsed_targets]

    new_pairs = list(pairs)
    new_targets = list(targets)

    for idx in replace_idx:
        # Replace target with sample from collapsed distribution
        new_target = rng.choice(collapsed_targets, p=collapsed_weights)
        new_targets[idx] = int(new_target)
        # Optionally also corrupt the pair (simulating input collapse)
        # For now, keep inputs clean — only corrupt outputs

    return new_pairs, new_targets""",
"""def apply_collapse(
    pairs: list, targets: list, prime: int,
    collapse_level: float, collapse_severity: float, seed: int,
    tree_depth: int = 1
) -> Tuple[list, list]:
    \"\"\"
    Simulate model collapse by replacing some targets with outputs from a "collapsed" model.
    \"\"\"
    new_pairs = list(pairs)
    new_targets = list(targets)

    # We use round() instead of int() to prevent silent data truncation biases
    n_replace = int(round(len(targets) * collapse_level))
    if n_replace == 0:
        return new_pairs, new_targets

    for depth in range(tree_depth):
        # Derive generation-specific seed to ensure each generation is reproducible
        # We use a 32-bit compatible hash
        gen_seed = hash((seed, "collapse_generation", depth)) % (2**32)
        rng = np.random.RandomState(gen_seed)

        replace_idx = rng.choice(len(new_targets), n_replace, replace=False)

        # Compute target frequency distribution
        from collections import Counter
        target_counts = Counter(new_targets)
        total = len(new_targets)

        # Explicit sorting for cross-environment determinism
        sorted_keys = sorted(target_counts.keys())
        freq = {t: target_counts[t] / total for t in sorted_keys}

        # Create collapsed distribution: amplify common targets, suppress rare ones
        temp = max(0.1, 1.0 - collapse_severity)
        collapsed_probs = {}
        for t in range(prime):
            base_prob = freq.get(t, 1.0 / prime)
            collapsed_probs[t] = base_prob ** (1.0 / temp)

        # Normalize
        total_prob = sum(collapsed_probs.values())
        sorted_targets = sorted(collapsed_probs.keys())
        collapsed_probs = {t: collapsed_probs[t] / total_prob for t in sorted_targets}

        # Allocate exact integer counts per target to prevent distribution drift
        counts = {t: int(np.floor(collapsed_probs[t] * n_replace)) for t in sorted_targets}
        remainder = n_replace - sum(counts.values())
        if remainder > 0:
            fracs = {t: collapsed_probs[t] * n_replace - counts[t] for t in sorted_targets}
            sorted_by_frac = sorted(sorted_targets, key=lambda k: (-fracs[k], k))
            for k in sorted_by_frac[:remainder]:
                counts[k] += 1

        # Generate exact samples list and shuffle deterministically
        samples = []
        for t in sorted_targets:
            samples.extend([t] * counts[t])
        rng.shuffle(samples)

        for idx, new_target in zip(replace_idx, samples):
            new_targets[idx] = int(new_target)

    return new_pairs, new_targets""")

with open("src/data.py", "w") as f:
    f.write(content)
