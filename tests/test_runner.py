import torch
import torch.nn as nn
from src.experiments.config import ExperimentConfig, CollapseConfig
from src.experiments.runner import ExperimentRunner

def test_collapse_injection():
    config = ExperimentConfig(epochs=1)
    runner = ExperimentRunner(config)

    # We'll use a simple linear layer to verify noise injection
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10, bias=False)
            nn.init.constant_(self.linear.weight, 1.0)

    model = SimpleModel()

    # Baseline weights
    initial_weight_sum = model.linear.weight.sum().item()

    # Inject collapse
    collapse_config = CollapseConfig(collapse_type="weight_noise", severity="severe", injection_point="model")
    runner.inject_collapse(model, collapse_config)

    # Weights should change
    new_weight_sum = model.linear.weight.sum().item()
    assert initial_weight_sum != new_weight_sum

def test_runner_initialization():
    config = ExperimentConfig(epochs=1)
    runner = ExperimentRunner(config)
    assert runner.model is not None
    assert runner.train_loader is not None
    assert runner.test_loader is not None
