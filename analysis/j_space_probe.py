import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Tuple

def get_jacobian(model, prime: int, layer_idx: int = -1, device: str = 'cpu') -> torch.Tensor:
    """
    Computes the Jacobian of the expected outputs w.r.t residual stream activations at a specific layer.
    For this 1-layer transformer, layer_idx=-1 means before the output head, where it's exactly the
    output head weights. For layer_idx=0 (before the first transformer layer), we use autograd.
    """
    model.eval()
    model.to(device)

    if layer_idx == -1:
        # Before output head
        return model.output_head.weight.detach().clone() # (prime, d_model)
    else:
        # We need the full Jacobian of logits w.r.t. the residual stream at layer layer_idx
        # For simplicity, we just use autograd on a subset of inputs

        # We define a function from h (batch, seq, d_model) -> logits (batch, prime)
        def forward_from_layer(h):
            # h is (batch, seq, d_model)
            h = model.transformer(h)
            h = model.ln(h)
            h = h.mean(dim=1)
            logits = model.output_head(h)
            return logits

        # We evaluate at a random dummy h
        h_dummy = torch.randn(1, 2, model.d_model, device=device, requires_grad=True)
        # Compute Jacobian: (1, prime, 1, 2, d_model)
        J = torch.autograd.functional.jacobian(forward_from_layer, h_dummy)

        # We take the mean across the sequence dimension and batch dimensions to get a (prime, d_model) approx
        J_approx = J.squeeze(0).squeeze(1).mean(dim=1) # (prime, d_model)
        return J_approx.detach()

def get_j_space_svd(model, prime: int, layer_idx: int = -1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs SVD on the Jacobian matrix to identify high-variance (J-space) directions.
    Returns:
        U, S, Vh
    """
    J = get_jacobian(model, prime, layer_idx, device)
    # Perform SVD
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    return U, S, Vh

def causal_intervention_j_space(model, prime: int, device: str = 'cpu') -> Dict[str, float]:
    """
    Edits J-space activations by projecting residual stream out of top J-space directions
    and measures accuracy impact.
    """
    model.eval()
    model.to(device)

    # Get J-space SVD
    _, S, Vh = get_j_space_svd(model, prime, -1, device)

    # Get standard accuracy
    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    x = torch.tensor(all_pairs, device=device)

    # We support addition task as default, but you might want to dynamically pass target logic if needed
    # For now we use standard modular addition (a + b) % prime as in src/data.py
    targets = torch.tensor([(a + b) % prime for a, b in all_pairs], device=device)

    with torch.no_grad():
        base_logits = model(x)
        base_preds = base_logits.argmax(dim=-1)
        base_acc = (base_preds == targets).float().mean().item()

        # Intervene: remove top k directions
        # h = tok + pos ... -> transformer -> pool -> output_head
        # We need a hook to modify the activation before output_head

        # Top 5 directions
        k = 5
        top_k_dirs = Vh[:k] # (k, d_model)

        # Projection matrix to remove top k directions
        # P = I - V_k^T V_k
        I = torch.eye(model.d_model, device=device)
        P = I - torch.matmul(top_k_dirs.T, top_k_dirs)

        # Register hook
        def hook(module, input):
            # input to linear is a tuple (h,)
            h = input[0]
            # project
            h_proj = torch.matmul(h, P)
            return (h_proj,)

        handle = model.output_head.register_forward_pre_hook(hook)

        interv_logits = model(x)
        interv_preds = interv_logits.argmax(dim=-1)
        interv_acc = (interv_preds == targets).float().mean().item()

        handle.remove()

    return {
        'base_acc': base_acc,
        'interv_acc_top_5_removed': interv_acc
    }

def compare_j_space(run1_dir: str, run2_dir: str, model_cls, prime: int, output_dir: str):
    """
    Compares J-space structure between grokked and collapsed models.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model1 = model_cls(prime=prime)
    model2 = model_cls(prime=prime)

    ckpts1 = sorted(glob.glob(os.path.join(run1_dir, "checkpoint_*.pt")), key=os.path.getmtime)
    ckpts2 = sorted(glob.glob(os.path.join(run2_dir, "checkpoint_*.pt")), key=os.path.getmtime)

    if not ckpts1 or not ckpts2:
        return

    ckpt1 = torch.load(ckpts1[-1], map_location=device)
    model1.load_state_dict(ckpt1['model_state'])

    ckpt2 = torch.load(ckpts2[-1], map_location=device)
    model2.load_state_dict(ckpt2['model_state'])

    _, S1, _ = get_j_space_svd(model1, prime, device)
    _, S2, _ = get_j_space_svd(model2, prime, device)

    plt.figure(figsize=(8, 5))
    plt.plot(S1.cpu().numpy(), label='Grokked Model', marker='o')
    plt.plot(S2.cpu().numpy(), label='Collapsed Model', marker='x')
    plt.title('J-Space Singular Values')
    plt.xlabel('Component index')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'j_space_svd_comparison.png'))
    plt.close()
