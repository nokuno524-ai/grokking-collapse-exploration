import torch
import numpy as np
from scipy import stats
from typing import Tuple, List, Callable


def detect_subliminal_traits(
    teacher_outputs: torch.Tensor,
    student_model: torch.nn.Module,
    sae_features: torch.Tensor,
) -> float:
    """
    Test if training on teacher outputs transmits traits not present in the explicit content.
    Calculates the correlation/projection between the empirical teacher outputs and the
    student model embeddings projected onto SAE features.
    """
    with torch.no_grad():
        # (prime, d_model)
        embeddings = student_model.token_embed.weight.detach()

        # Calculate projection of embeddings onto sae_features
        # sae_features assumed to be (d_model, n_features)
        projections = torch.matmul(embeddings, sae_features) # (prime, n_features)

        # If teacher outputs are provided (assumed shape: batch_size x prime),
        # compute correlation of mean outputs to feature projections to detect
        # if the outputs transmit the trait
        teacher_mean = teacher_outputs.mean(dim=0) # (prime,)
        teacher_mean = teacher_mean / (teacher_mean.norm() + 1e-10)

        proj_mean = projections.mean(dim=-1) # (prime,)
        proj_mean = proj_mean / (proj_mean.norm() + 1e-10)

        # Return correlation (cosine similarity)
        return torch.dot(teacher_mean, proj_mean).abs().item()


def measure_trait_transfer(
    control_data: torch.Tensor,
    teacher_data: torch.Tensor,
    model: torch.nn.Module,
) -> float:
    """
    Compare models trained on semantically identical but stylistically different data.
    """
    with torch.no_grad():
        # Evaluate model on control and teacher data
        # Assume data is (batch, 2) input for ModularArithmeticTransformer
        control_out = model(control_data)
        teacher_out = model(teacher_data)

        # Transfer is measured as the difference in output variance/style
        transfer_score = (teacher_out.var(dim=-1).mean() - control_out.var(dim=-1).mean()).abs().item()
        return transfer_score


def trait_projection_score(
    model: torch.nn.Module,
    trait_direction: torch.Tensor,
) -> float:
    """
    Project model activations onto a trait direction.
    """
    with torch.no_grad():
        embeddings = model.token_embed.weight.detach() # (prime, d_model)

        # Normalize trait direction
        trait_dir_norm = trait_direction / (torch.norm(trait_direction) + 1e-10)

        # Project
        projection = torch.matmul(embeddings, trait_dir_norm) # (prime,)
        return projection.abs().mean().item()


def bootstrap_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals to test statistical significance.
    """
    # Ensure data is 1D
    data = np.asarray(data).flatten()

    # Avoid degenerate variance errors by adding slight noise if all values are identical
    if np.var(data) == 0:
        data = data + np.random.normal(0, 1e-10, size=data.shape)

    res = stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=confidence_level,
        method='bca',
        n_resamples=1000,
        axis=-1
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)
