import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

class MechInterpSuite:
    """
    Mechanistic Interpretability tooling.
    Tests for functional welfare axis from ICML 2026.
    Extracts steering vectors, performs causal patching, and analyzes attention patterns.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def fit_linear_probe(self, hidden_states: np.ndarray, labels: np.ndarray) -> float:
        """
        Linear probe fitting: can we decode "task accuracy" from hidden states?
        Tests for functional welfare axis from ICML 2026.

        Args:
            hidden_states: (N, D) array of hidden representations.
            labels: (N,) array of binary labels (e.g. 1 for correct, 0 for incorrect).

        Returns:
            Accuracy of the linear probe.
        """
        if len(np.unique(labels)) < 2:
            return 1.0 # Trivial if only one class

        probe = LogisticRegression(max_iter=1000)
        probe.fit(hidden_states, labels)
        score = probe.score(hidden_states, labels)
        return float(score)

    def extract_steering_vector(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Steering vector extraction: compute difference-of-means between
        correct and incorrect output hidden states.

        Args:
            hidden_states: (N, D) tensor.
            labels: (N,) binary tensor (1=correct, 0=incorrect).

        Returns:
            (D,) tensor representing the steering direction.
        """
        correct_mask = (labels == 1)
        incorrect_mask = (labels == 0)

        if not correct_mask.any() or not incorrect_mask.any():
            return torch.zeros(hidden_states.size(1), device=self.device)

        mean_correct = hidden_states[correct_mask].mean(dim=0)
        mean_incorrect = hidden_states[incorrect_mask].mean(dim=0)

        return mean_correct - mean_incorrect

    def causal_patching(self, inputs: torch.Tensor, source_hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Causal patching: swap activations between conditions (e.g. clean and collapse)
        to identify which layers carry the "collapse signal".

        Args:
            inputs: The input tokens for the forward pass.
            source_hidden: The hidden state to patch from.
            layer_idx: Which layer we are patching.

        Returns:
            The model logits after causal patching.
        """
        handle = None

        def patch_hook(module, args, output):
            # output might be a tuple
            if isinstance(output, tuple):
                return (source_hidden,) + output[1:]
            return source_hidden

        try:
            # target the specific layer in ModularArithmeticTransformer
            layer = self.model.transformer.layers[layer_idx]
            handle = layer.register_forward_hook(patch_hook)
            with torch.no_grad():
                logits = self.model(inputs)
            return logits
        finally:
            if handle is not None:
                handle.remove()

    def analyze_attention_patterns(self, attn_weights: torch.Tensor, inputs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Attention pattern analysis: detect head specialization
        - Induction heads
        - Duplicate token heads
        - Previous token heads

        Args:
            attn_weights: Tensor of shape (batch, n_heads, seq_len, seq_len)
            inputs: Original input tokens of shape (batch, seq_len)

        Returns:
            Dictionary mapping pattern type to score.
        """
        batch_size, n_heads, seq_len, _ = attn_weights.shape
        scores = {
            "induction_head_score": 0.0,
            "duplicate_token_score": 0.0,
            "previous_token_score": 0.0
        }

        if seq_len < 2:
            return scores

        # Previous token score: attention to the immediately preceding token
        prev_tok_attn = 0.0
        for i in range(1, seq_len):
            prev_tok_attn += attn_weights[:, :, i, i-1].mean().item()
        scores["previous_token_score"] = prev_tok_attn / (seq_len - 1)

        # We compute proxy scores for induction and duplicate token heads if inputs are missing.
        # But if inputs are provided, we compute them precisely.
        if inputs is None:
            return scores

        # duplicate token score: attention to previous occurrences of the current token
        dup_score = 0.0
        total_dups = 0
        for b in range(batch_size):
            for i in range(1, seq_len):
                current_tok = inputs[b, i].item()
                for j in range(i):
                    if inputs[b, j].item() == current_tok:
                        dup_score += attn_weights[b, :, i, j].mean().item()
                        total_dups += 1

        if total_dups > 0:
            scores["duplicate_token_score"] = dup_score / total_dups

        # induction head score: attention to the token immediately following a previous occurrence of the current token
        ind_score = 0.0
        total_inds = 0
        for b in range(batch_size):
            for i in range(1, seq_len):
                current_tok = inputs[b, i].item()
                # look for A B ... A, where current token is A, and we want to see if it attends to B
                # The token we want to attend to is the one after the previous A
                for j in range(i - 1):
                    if inputs[b, j].item() == current_tok:
                        ind_score += attn_weights[b, :, i, j+1].mean().item()
                        total_inds += 1

        if total_inds > 0:
            scores["induction_head_score"] = ind_score / total_inds

        return scores
