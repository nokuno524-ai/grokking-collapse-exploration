import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict


def track_persona_features(
    model_checkpoints: List[torch.nn.Module],
    persona_directions: torch.Tensor
) -> np.ndarray:
    """
    Project checkpoint weights onto persona_directions.
    Returns trajectory of projections over time.
    """
    trajectories = []

    # Ensure directions are normalized
    persona_directions = persona_directions / (torch.norm(persona_directions, dim=0, keepdim=True) + 1e-10)

    with torch.no_grad():
        for model in model_checkpoints:
            # We track the token embedding weights as proxies for features
            weights = model.token_embed.weight.detach() # (prime, d_model)

            # Project weights onto persona directions
            # Assume persona_directions is (d_model, n_directions)
            projection = torch.matmul(weights, persona_directions) # (prime, n_directions)

            # Store the mean magnitude of projections across prime tokens
            proj_mag = projection.abs().mean(dim=0).cpu().numpy()
            trajectories.append(proj_mag)

    return np.array(trajectories) # (n_checkpoints, n_directions)


def compute_persona_stability(feature_trajectories: np.ndarray) -> np.ndarray:
    """
    Compute variance of persona features over time.
    Lower variance indicates higher stability.
    """
    # Calculate standard deviation along the time axis (checkpoints)
    stability = np.std(feature_trajectories, axis=0)
    return stability


def plot_persona_trajectories(
    trajectories_collapsed: np.ndarray,
    trajectories_pure: np.ndarray,
    filename: str
):
    """
    Visualize feature changes comparing collapsed vs pure models.
    """
    plt.figure(figsize=(10, 6))

    # Plot average trajectory across all persona directions
    mean_collapsed = trajectories_collapsed.mean(axis=1)
    std_collapsed = trajectories_collapsed.std(axis=1)

    mean_pure = trajectories_pure.mean(axis=1)
    std_pure = trajectories_pure.std(axis=1)

    epochs = np.arange(len(mean_collapsed))

    plt.plot(epochs, mean_collapsed, label='Collapsed', color='red')
    plt.fill_between(epochs, mean_collapsed - std_collapsed, mean_collapsed + std_collapsed, color='red', alpha=0.2)

    plt.plot(epochs, mean_pure, label='Pure (Non-collapsed)', color='blue')
    plt.fill_between(epochs, mean_pure - std_pure, mean_pure + std_pure, color='blue', alpha=0.2)

    plt.title('Persona Persistence: Feature Trajectories')
    plt.xlabel('Training Steps / Checkpoints')
    plt.ylabel('Persona Feature Projection Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
