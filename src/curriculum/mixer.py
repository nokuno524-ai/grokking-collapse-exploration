"""
Data mixer for combining datasets according to a schedule.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import Dataset, TensorDataset
from src.curriculum.schedules import Schedule, ConstantSchedule


class DataMixer:
    """
    Mixes a 'fresh' dataset and a 'collapsed' dataset dynamically based on a schedule.
    Since we need to pull from both distributions without just repeating exactly,
    we sample indices from the given datasets.
    """
    def __init__(
        self,
        fresh_inputs: torch.Tensor,
        fresh_targets: torch.Tensor,
        collapsed_inputs: torch.Tensor,
        collapsed_targets: torch.Tensor,
        schedule: Schedule,
        batch_size: int,
        seed: int = 42
    ):
        self.fresh_inputs = fresh_inputs
        self.fresh_targets = fresh_targets
        self.collapsed_inputs = collapsed_inputs
        self.collapsed_targets = collapsed_targets
        self.schedule = schedule
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Validation
        assert len(fresh_inputs) == len(fresh_targets)
        assert len(collapsed_inputs) == len(collapsed_targets)
        self.n_fresh = len(fresh_inputs)
        self.n_collapsed = len(collapsed_inputs)

    def get_batch(self, step: int, max_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a mixed batch for the current step.
        """
        fresh_fraction = self.schedule.get_fresh_fraction(step, max_steps)
        n_fresh_in_batch = int(round(self.batch_size * fresh_fraction))
        n_collapsed_in_batch = self.batch_size - n_fresh_in_batch

        batch_inputs = []
        batch_targets = []

        if n_fresh_in_batch > 0:
            idx_fresh = self.rng.choice(self.n_fresh, n_fresh_in_batch, replace=True)
            batch_inputs.append(self.fresh_inputs[idx_fresh])
            batch_targets.append(self.fresh_targets[idx_fresh])

        if n_collapsed_in_batch > 0:
            idx_collapsed = self.rng.choice(self.n_collapsed, n_collapsed_in_batch, replace=True)
            batch_inputs.append(self.collapsed_inputs[idx_collapsed])
            batch_targets.append(self.collapsed_targets[idx_collapsed])

        if len(batch_inputs) == 0:
            # Fallback for weird edge cases (e.g. batch size 0)
            return torch.empty((0, self.fresh_inputs.shape[1]), dtype=self.fresh_inputs.dtype), torch.empty((0,), dtype=self.fresh_targets.dtype)

        # Concatenate and shuffle within the batch
        b_in = torch.cat(batch_inputs, dim=0)
        b_tgt = torch.cat(batch_targets, dim=0)

        # Shuffle the batch so the order of fresh vs collapsed is random
        shuffle_idx = torch.randperm(self.batch_size)
        return b_in[shuffle_idx], b_tgt[shuffle_idx]
