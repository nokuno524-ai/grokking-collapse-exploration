def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

old_track = """def track_gradient_norms(checkpoints: List[Dict]) -> List[float]:
    \"\"\"
    Tracks overall gradient norms over training using weight differences.

    Args:
        checkpoints: List of model state dicts.

    Returns:
        List of gradient norms. Length will be len(checkpoints) - 1.
    \"\"\"
    from src.model import ModularArithmeticTransformer

    if len(checkpoints) < 2:
        return []

    norms = []

    model_prev = ModularArithmeticTransformer()
    model_curr = ModularArithmeticTransformer()

    for i in range(1, len(checkpoints)):
        model_prev.load_state_dict(checkpoints[i-1]['model_state'])
        model_curr.load_state_dict(checkpoints[i]['model_state'])

        grads = approximate_gradients(model_prev, model_curr)

        total_norm = 0.0
        for grad in grads.values():
            total_norm += torch.norm(grad, p=2).item() ** 2

        norms.append(total_norm ** 0.5)

    return norms"""

new_track = """def track_gradient_norms(checkpoints: List[Dict]) -> Dict[str, List[float]]:
    \"\"\"
    Tracks gradient norms per layer over training using weight differences.

    Args:
        checkpoints: List of model state dicts.

    Returns:
        Dictionary mapping parameter names to lists of gradient norms over time.
    \"\"\"
    from src.model import ModularArithmeticTransformer
    from collections import defaultdict

    if len(checkpoints) < 2:
        return {}

    norms_per_layer = defaultdict(list)

    model_prev = ModularArithmeticTransformer()
    model_curr = ModularArithmeticTransformer()

    for i in range(1, len(checkpoints)):
        model_prev.load_state_dict(checkpoints[i-1]['model_state'])
        model_curr.load_state_dict(checkpoints[i]['model_state'])

        grads = approximate_gradients(model_prev, model_curr)

        for name, grad in grads.items():
            norms_per_layer[name].append(torch.norm(grad, p=2).item())

    return dict(norms_per_layer)"""

replace_in_file("analysis/gradient_flow.py", old_track, new_track)
