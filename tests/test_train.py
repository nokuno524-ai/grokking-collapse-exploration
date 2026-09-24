import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.train import train, TrainConfig, TrainState

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        # Return float tensor of shape (batch, prime) to act as logits
        # Must depend on a parameter so loss.backward() works
        return torch.randn(x.size(0), 59, dtype=torch.float32, requires_grad=True) * self.param

    def get_weight_norm(self):
        return 1.0

    def get_embedding_rank(self):
        return 1.0

    def get_embedding_fourier_spectrum(self):
        return torch.ones(59, 128)

def test_grokking_requires_sustained_accuracy():
    config = TrainConfig(
        max_steps=1000,
        eval_every=100,
        condition_name="test_sustained",
        output_dir="/tmp/results"
    )

    # We will mock 'evaluate' to return specific accuracies over time.
    # We eval every 100 steps. Max 1000 steps = 10 evaluations.
    # evaluate is called TWICE per eval (train_loader, test_loader)
    accuracies = [
        0.5, 0.5,     # eval 1 (step 100)
        0.98, 0.98,   # eval 2 (step 200)
        0.98, 0.98,   # eval 3 (step 300)
        0.99, 0.99,   # eval 4 (step 400 - spike!)
        0.98, 0.98,   # eval 5 (step 500 - dropped, resets counter)
        0.99, 0.99,   # eval 6 (step 600 - start sustained)
        0.99, 0.99,   # eval 7 (step 700 - 2)
        0.99, 0.99,   # eval 8 (step 800 - 3)
        0.99, 0.99,   # eval 9 (step 900 - 4)
        0.99, 0.99    # eval 10 (step 1000 - 5 - grok!)
    ]
    losses = [1.0] * 20
    eval_returns = list(zip(losses, accuracies))

    with patch('src.train.evaluate', side_effect=eval_returns), \
         patch('src.train.ModularArithmeticTransformer', return_value=MockModel()), \
         patch('src.train.generate_modular_arithmetic') as mock_gen, \
         patch('torch.save'): # Don't actually save

        # Mock dataloader inputs (dummy shapes)
        dummy_tensor = torch.zeros((10, 2), dtype=torch.long)
        dummy_target = torch.zeros((10,), dtype=torch.long)
        mock_gen.return_value = (dummy_tensor, dummy_target, dummy_tensor, dummy_target)

        state = train(config)

        assert state.grokked == True
        assert state.grokking_step == 1000

        history = state.history
        assert len(history) == 10
        assert history[3]['test_acc'] == 0.99