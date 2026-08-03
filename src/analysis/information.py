import torch
import numpy as np

def compute_entropy_binned(tensor: torch.Tensor, bins: int = 100) -> float:
    """
    Computes Shannon entropy of a tensor using histogram binning.
    tensor: expected to be a 1D tensor or flattened.
    """
    tensor_np = tensor.detach().cpu().numpy().flatten()
    hist, _ = np.histogram(tensor_np, bins=bins, density=False)

    # Convert to probabilities
    probs = hist / float(np.sum(hist))

    # filter out zeros for log computation
    probs = probs[probs > 0]

    # Calculate discrete entropy
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)

def compute_mutual_information_binned(x: torch.Tensor, y: torch.Tensor, bins: int = 20) -> float:
    """
    Computes mutual information between two tensors x and y using 2D histogram binning.
    Tensors will be flattened.
    """
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()

    if len(x_np) != len(y_np):
        # We need pairs. If shapes differ, we might need to handle differently.
        # Assuming they are feature vectors of same size, or we just sample them.
        min_len = min(len(x_np), len(y_np))
        x_np = x_np[:min_len]
        y_np = y_np[:min_len]

    hist_2d, _, _ = np.histogram2d(x_np, y_np, bins=bins)

    # Convert to probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Compute MI
    px_py = px[:, None] * py[None, :]

    # Mask to avoid log(0)
    nzs = pxy > 0

    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return float(mi)

def compute_information_flow(model: torch.nn.Module, inputs: torch.Tensor, logits: torch.Tensor) -> dict:
    """
    Computes information flow metrics for a ModularArithmeticTransformer.
    Requires intermediate representations.
    """
    # Forward pass again but explicitly extract intermediate values
    # Actually, it's better to hook the model or recompute here if simple
    # Since ModularArithmeticTransformer is simple:

    with torch.no_grad():
        tok = model.token_embed(inputs)
        positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions)

        h_0 = tok + pos

        # We also need h_1, output of transformer
        h_1 = model.transformer(h_0)

    metrics = {
        "entropy_h0": compute_entropy_binned(h_0),
        "entropy_h1": compute_entropy_binned(h_1),
        "mi_input_h0": compute_mutual_information_binned(inputs.float(), h_0),
        "mi_h0_h1": compute_mutual_information_binned(h_0, h_1),
        "mi_h1_output": compute_mutual_information_binned(h_1, logits)
    }

    return metrics
