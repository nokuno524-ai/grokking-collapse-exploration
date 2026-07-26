import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.model import ModularArithmeticTransformer

def get_effective_rank(W):
    s = torch.linalg.svdvals(W)
    s = s / s.sum()
    entropy = -(s * torch.log(s + 1e-10)).sum()
    return torch.exp(entropy).item()

def get_singular_values(W):
    return torch.linalg.svdvals(W).detach().numpy()

def compute_neuron_selectivity(model, x):
    """
    How monosemantic are the neurons? We compute activations across inputs and
    measure the skewness/kurtosis or sparsity of activations.
    """
    batch_size = x.shape[0]

    # Forward pass to get activations
    tok = model.token_embed(x)
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    layer = model.transformer.layers[0]
    h2 = layer.self_attn(h, h, h)[0]
    h = h + layer.dropout1(h2)
    h = layer.norm1(h)

    ff_out1 = layer.linear1(h)
    activations = F.gelu(ff_out1) # (batch, seq_len, d_ff)
    activations = activations.mean(dim=1) # average over seq_len

    act = activations.detach().numpy()
    gini_coeffs = []
    for i in range(act.shape[1]):
        neuron_act = act[:, i]
        neuron_act = np.abs(neuron_act) + 1e-8
        sorted_act = np.sort(neuron_act)
        n = len(neuron_act)
        cum_act = np.cumsum(sorted_act)
        gini = (n + 1 - 2 * np.sum(cum_act) / cum_act[-1]) / n
        gini_coeffs.append(gini)

    return np.mean(gini_coeffs), gini_coeffs

def track_gradient_noise_and_transition(model, x, target):
    """
    Measures the gradient noise scale/norm by executing a backward pass.
    A sharp drop in gradient norm indicates the grokking transition.
    """
    model.train()
    model.zero_grad()

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    model.eval()
    return total_norm

def main():
    os.makedirs("results/circuits", exist_ok=True)

    # Using pure condition
    steps = list(range(5000, 50001, 5000))

    prime = 59
    a = torch.arange(prime)
    b = torch.arange(prime)
    grid_a, grid_b = torch.meshgrid(a, b, indexing='ij')
    x_test = torch.stack([grid_a.flatten(), grid_b.flatten()], dim=-1)
    targets = (grid_a.flatten() + grid_b.flatten()) % prime

    ranks = []
    selectivities = []
    grad_norms = []

    for step in steps:
        ckpt_path = f"results/pure/checkpoint_{step}.pt"
        if not os.path.exists(ckpt_path):
            continue

        model = ModularArithmeticTransformer()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)

        # Track gradient norm (noise) and transition point
        gnorm = track_gradient_noise_and_transition(model, x_test, targets)
        grad_norms.append((step, gnorm))

        model.eval()
        with torch.no_grad():
            W_embed = model.token_embed.weight
            W_out = model.output_head.weight

            rank_embed = get_effective_rank(W_embed)
            rank_out = get_effective_rank(W_out)
            ranks.append((step, rank_embed, rank_out))

            mean_gini, _ = compute_neuron_selectivity(model, x_test)
            selectivities.append((step, mean_gini))

            if step in [5000, 15000, 50000]:
                s_embed = get_singular_values(W_embed)
                s_out = get_singular_values(W_out)

                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(s_embed)
                plt.title(f"Embed Singular Values (Step {step})")
                plt.yscale('log')

                plt.subplot(1, 2, 2)
                plt.plot(s_out)
                plt.title(f"Output Singular Values (Step {step})")
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(f"results/circuits/svd_step_{step}.png")
                plt.close()

    steps_arr = [r[0] for r in ranks]

    # 1. Rank evolution
    rank_embed_arr = [r[1] for r in ranks]
    rank_out_arr = [r[2] for r in ranks]
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, rank_embed_arr, label="Embedding Rank", marker='o')
    plt.plot(steps_arr, rank_out_arr, label="Output Rank", marker='o')
    plt.xlabel("Training Step")
    plt.ylabel("Effective Rank")
    plt.title("Weight Matrix Rank Evolution (Pure)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/circuits/rank_evolution.png")
    plt.close()

    # 2. Selectivity evolution
    sel_arr = [s[1] for s in selectivities]
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, sel_arr, marker='o', color='purple')
    plt.xlabel("Training Step")
    plt.ylabel("Mean Gini Coefficient")
    plt.title("Neuron Selectivity Evolution (Pure)")
    plt.grid(True)
    plt.savefig("results/circuits/neuron_selectivity.png")
    plt.close()

    # 3. Gradient Norm (Transition Detection)
    gnorm_arr = [g[1] for g in grad_norms]

    # Detect transition: argmin of gradient diff, or simply where it drops sharply
    diffs = np.diff(gnorm_arr)
    # the step before the sharpest drop
    transition_idx = np.argmin(diffs)
    transition_step = steps_arr[transition_idx]

    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, gnorm_arr, marker='o', color='red')
    plt.axvline(x=transition_step, color='black', linestyle='--', label=f'Grokking Transition (Gradient Drop)\nStep ~{transition_step}')
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm / Noise Scale Evolution (Pure)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/circuits/gradient_norm_transition.png")
    plt.close()

if __name__ == "__main__":
    main()
