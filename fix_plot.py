def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

old_plot = """def plot_gradient_cosine_similarity(checkpoints: List[Dict]) -> List[float]:
    \"\"\"
    Computes gradient cosine similarity between consecutive steps.

    Args:
        checkpoints: List of model state dicts.

    Returns:
        List of cosine similarities. Length will be len(checkpoints) - 2.
    \"\"\"
    from src.model import ModularArithmeticTransformer

    if len(checkpoints) < 3:
        return []

    sims = []

    model_t0 = ModularArithmeticTransformer()
    model_t1 = ModularArithmeticTransformer()
    model_t2 = ModularArithmeticTransformer()

    for i in range(2, len(checkpoints)):
        model_t0.load_state_dict(checkpoints[i-2]['model_state'])
        model_t1.load_state_dict(checkpoints[i-1]['model_state'])
        model_t2.load_state_dict(checkpoints[i]['model_state'])

        grads_1 = approximate_gradients(model_t0, model_t1)
        grads_2 = approximate_gradients(model_t1, model_t2)

        flat_1 = torch.cat([g.contiguous().view(-1) for g in grads_1.values()])
        flat_2 = torch.cat([g.contiguous().view(-1) for g in grads_2.values()])

        if torch.norm(flat_1) == 0 or torch.norm(flat_2) == 0:
            sims.append(0.0)
        else:
            sim = torch.nn.functional.cosine_similarity(flat_1, flat_2, dim=0)
            sims.append(sim.item())

    return sims"""

new_plot = """def plot_gradient_cosine_similarity(checkpoints: List[Dict], output_path: str = None) -> List[float]:
    \"\"\"
    Computes and plots gradient cosine similarity between consecutive steps.

    Args:
        checkpoints: List of model state dicts.
        output_path: Optional path to save the plot.

    Returns:
        List of cosine similarities. Length will be len(checkpoints) - 2.
    \"\"\"
    from src.model import ModularArithmeticTransformer
    import matplotlib.pyplot as plt

    if len(checkpoints) < 3:
        return []

    sims = []

    model_t0 = ModularArithmeticTransformer()
    model_t1 = ModularArithmeticTransformer()
    model_t2 = ModularArithmeticTransformer()

    for i in range(2, len(checkpoints)):
        model_t0.load_state_dict(checkpoints[i-2]['model_state'])
        model_t1.load_state_dict(checkpoints[i-1]['model_state'])
        model_t2.load_state_dict(checkpoints[i]['model_state'])

        grads_1 = approximate_gradients(model_t0, model_t1)
        grads_2 = approximate_gradients(model_t1, model_t2)

        flat_1 = torch.cat([g.contiguous().view(-1) for g in grads_1.values()])
        flat_2 = torch.cat([g.contiguous().view(-1) for g in grads_2.values()])

        if torch.norm(flat_1) == 0 or torch.norm(flat_2) == 0:
            sims.append(0.0)
        else:
            sim = torch.nn.functional.cosine_similarity(flat_1, flat_2, dim=0)
            sims.append(sim.item())

    if output_path:
        plt.figure(figsize=(10, 6))
        plt.plot(sims, marker='o')
        plt.title('Gradient Cosine Similarity Between Steps')
        plt.xlabel('Step Index')
        plt.ylabel('Cosine Similarity')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    return sims"""

replace_in_file("analysis/gradient_flow.py", old_plot, new_plot)
