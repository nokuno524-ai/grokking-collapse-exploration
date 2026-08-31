import torch
import numpy as np

def generate_curriculum_batch(
    clean_in: torch.Tensor,
    clean_tgt: torch.Tensor,
    collapse_in: torch.Tensor,
    collapse_tgt: torch.Tensor,
    batch_size: int,
    w: float,
    rng: np.random.RandomState
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a mixed batch from clean and collapsed datasets.
    w is the probability of choosing a collapsed example.
    """
    assert len(clean_in) == len(clean_tgt)
    assert len(collapse_in) == len(collapse_tgt)

    n_collapse = rng.binomial(batch_size, w)
    n_clean = batch_size - n_collapse

    if n_clean > 0:
        clean_idx = rng.choice(len(clean_in), n_clean, replace=True)
        batch_clean_in = clean_in[clean_idx]
        batch_clean_tgt = clean_tgt[clean_idx]
    else:
        batch_clean_in = torch.empty((0, *clean_in.shape[1:]), dtype=clean_in.dtype)
        batch_clean_tgt = torch.empty((0,), dtype=clean_tgt.dtype)

    if n_collapse > 0:
        collapse_idx = rng.choice(len(collapse_in), n_collapse, replace=True)
        batch_collapse_in = collapse_in[collapse_idx]
        batch_collapse_tgt = collapse_tgt[collapse_idx]
    else:
        batch_collapse_in = torch.empty((0, *collapse_in.shape[1:]), dtype=collapse_in.dtype)
        batch_collapse_tgt = torch.empty((0,), dtype=collapse_tgt.dtype)

    batch_in = torch.cat([batch_clean_in, batch_collapse_in], dim=0)
    batch_tgt = torch.cat([batch_clean_tgt, batch_collapse_tgt], dim=0)

    # Shuffle the batch
    shuffle_idx = torch.randperm(batch_size)

    return batch_in[shuffle_idx], batch_tgt[shuffle_idx]
