import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from src.data import DatasetConfig, generate_modular_arithmetic

class SyntheticDataGenerator:
    """
    Generates synthetic datasets with controlled levels of collapse and
    simulates sequence generation artifacts like temperature and top-k sampling.
    """

    def __init__(self, prime: int = 59, seed: int = 42):
        self.prime = prime
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _apply_sampling_strategy(self,
                                 targets: List[int],
                                 strategy: str,
                                 kwargs: Dict[str, Any]) -> List[int]:
        """
        Apply a sampling strategy to the targets to simulate generated data.
        """
        if strategy == "none" or strategy is None:
            return targets

        new_targets = list(targets)
        freq = np.bincount(targets, minlength=self.prime)
        total = len(targets)
        probs = freq / total

        # Add small epsilon for log
        epsilon = 1e-10
        logits = np.log(probs + epsilon)

        for i in range(len(new_targets)):
            # If we decide to sample instead of keeping the target
            # For simplicity in this dummy simulation, we just replace all targets
            # based on the strategy applied to the empirical distribution

            if strategy == "temperature":
                temp = kwargs.get("temperature", 1.0)
                scaled_logits = logits / temp
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                p = exp_logits / np.sum(exp_logits)
                new_targets[i] = self.rng.choice(self.prime, p=p)

            elif strategy == "top_k":
                k = kwargs.get("k", 5)
                top_k_idx = np.argsort(logits)[-k:]
                top_k_logits = logits[top_k_idx]
                exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                p = exp_logits / np.sum(exp_logits)
                new_targets[i] = self.rng.choice(top_k_idx, p=p)

            elif strategy == "nucleus":
                p_thresh = kwargs.get("p", 0.9)
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumulative_probs = np.cumsum(sorted_probs)

                # Find indices to keep
                keep_idx = sorted_idx[cumulative_probs <= p_thresh]
                if len(keep_idx) == 0:
                    keep_idx = sorted_idx[:1] # Keep at least one

                keep_probs = probs[keep_idx]
                keep_probs = keep_probs / np.sum(keep_probs)
                new_targets[i] = self.rng.choice(keep_idx, p=keep_probs)

        return new_targets

    def generate(self,
                 collapse_level: float = 0.0,
                 collapse_severity: float = 0.5,
                 sampling_strategy: str = "none",
                 sampling_kwargs: Dict[str, Any] = None) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate datasets at controlled collapse levels.

        Args:
            collapse_level: Fraction of training data replaced by synthetic (0.0 to 1.0)
            collapse_severity: How collapsed the synthetic generator is (0.0 to 1.0)
            sampling_strategy: "none", "temperature", "top_k", or "nucleus"
            sampling_kwargs: Arguments for the sampling strategy

        Returns:
            Tuple of (original_data, synthetic_data) where each is a list of sequences [a, b, target]
        """
        if sampling_kwargs is None:
            sampling_kwargs = {}

        # Get base dataset
        config = DatasetConfig(prime=self.prime,
                               collapse_level=collapse_level,
                               collapse_severity=collapse_severity,
                               seed=self.seed)

        train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

        # Convert to list of sequences [a, b, c]
        sequences = []
        for i in range(len(train_in)):
            seq = [train_in[i][0].item(), train_in[i][1].item(), train_tgt[i].item()]
            sequences.append(seq)

        # To identify synthetic data, we can re-generate pure data and compare,
        # or just split it. Since generate_modular_arithmetic applies collapse to a random subset,
        # we can just manually split here to simulate the clean/synthetic split.

        # Generate pure data
        pure_config = DatasetConfig(prime=self.prime, collapse_level=0.0, seed=self.seed)
        pure_in, pure_tgt, _, _ = generate_modular_arithmetic(pure_config)

        pure_sequences = []
        for i in range(len(pure_in)):
            seq = [pure_in[i][0].item(), pure_in[i][1].item(), pure_tgt[i].item()]
            pure_sequences.append(seq)

        n_synthetic = int(len(pure_sequences) * collapse_level)

        if n_synthetic == 0:
            return pure_sequences, []

        # Split into original and synthetic parts
        original_data = pure_sequences[:-n_synthetic]

        # The synthetic data will be based on the last n_synthetic pure sequences
        # but corrupted by the sampling strategy and severity
        base_synthetic = pure_sequences[-n_synthetic:]
        base_targets = [seq[2] for seq in base_synthetic]

        # First apply the sampling strategy
        sampled_targets = self._apply_sampling_strategy(base_targets, sampling_strategy, sampling_kwargs)

        # Then apply collapse severity (suppressing rare targets)
        # We reuse apply_collapse from src.data indirectly or re-implement simply here
        freq = np.bincount(sampled_targets, minlength=self.prime)
        total = len(sampled_targets)
        probs = freq / max(total, 1)

        temp = max(0.1, 1.0 - collapse_severity)
        collapsed_probs = np.power(probs, 1.0 / temp)
        collapsed_probs = collapsed_probs / np.sum(collapsed_probs)

        synthetic_data = []
        for i in range(len(base_synthetic)):
            a, b, _ = base_synthetic[i]
            target = self.rng.choice(self.prime, p=collapsed_probs)
            synthetic_data.append([a, b, int(target)])

        return original_data, synthetic_data
