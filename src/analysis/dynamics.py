import torch
import torch.nn as nn
from typing import Dict, Any

def track_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Track the gradient norm for each layer.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms

def compute_gradient_noise_scale(model: nn.Module, micro_batch_grads: Dict[str, torch.Tensor], full_batch_grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute gradient noise scale (batch-to-batch gradient variance).
    Returns the variance Tr(Cov(g)) scaled properly.
    (Simplified approximation: ||g_micro - g_full||^2)
    """
    noise_scales = {}
    for name in full_batch_grads.keys():
        if name in micro_batch_grads:
            diff = micro_batch_grads[name] - full_batch_grads[name]
            noise_scales[name] = (diff.norm() ** 2).item()
    return noise_scales

def estimate_hessian_eigenvalues(model: nn.Module, dataloader: Any, criterion: Any, device: torch.device, max_iters: int = 100, tol: float = 1e-3) -> float:
    """
    Estimate Hessian eigenvalues (loss landscape curvature) via power iteration.
    Detects if minima is flatter or sharper.
    """
    # Flash Attention must be explicitly disabled to prevent 'derivative not implemented' errors on CPU
    # When computing second derivatives on CPU, we need to disable flash attention via backends.
    # Note: `native_sdp_enable_flash` might not be enough on some torch versions.
    if device.type == 'cpu' or str(device) == 'cpu':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    model.eval()

    # Get a batch
    try:
        inputs, targets = next(iter(dataloader))
    except StopIteration:
        return 0.0

    inputs, targets = inputs.to(device), targets.to(device)

    # Calculate loss and grads
    logits = model(inputs)
    loss = criterion(logits, targets)

    params = [p for p in model.parameters() if p.requires_grad]

    # Compute first-order gradients
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Initialize random vector v
    v = [torch.randn_like(p) for p in params]

    # Normalize v
    v_norm = torch.sqrt(sum((x**2).sum() for x in v))
    v = [x / v_norm for x in v]

    eigenvalue = 0.0

    for i in range(max_iters):
        # Compute Hessian-vector product: H*v = d(g^T * v)/dw
        grad_v = sum(torch.sum(g * x) for g, x in zip(grads, v))

        # Backward pass on the dot product to get Hv
        Hv = torch.autograd.grad(grad_v, params, retain_graph=True)

        # Rayleigh quotient: v^T * H * v
        eigenvalue = sum(torch.sum(h * x).item() for h, x in zip(Hv, v))

        # Normalize new v
        v_norm = torch.sqrt(sum((h**2).sum() for h in Hv))
        if v_norm < 1e-8:
            break

        v_new = [h / v_norm for h in Hv]

        # Check convergence
        diff = sum(torch.sum((n - o)**2).item() for n, o in zip(v_new, v))
        if diff < tol:
            break

        v = v_new

    # Re-enable Flash attention flag if changed
    if device.type == 'cpu':
         torch.backends.native_sdp_enable_flash = True

    return eigenvalue
