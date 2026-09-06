import torch
import numpy as np
from src.analysis.hidden_state_probe import (
    extract_dataset_hidden_states,
    train_probes,
    run_probe_tracker
)
from src.model import ModularArithmeticTransformer

def test_train_probes():
    prime = 59
    # Dummy data
    inputs = np.array([[2, 3], [4, 4], [1, 0], [10, 50]])
    targets = np.array([5, 8, 1, 1])  # (10+50)%59 = 1

    # Dummy hidden states that perfectly separate parity_a
    # Dim 0 is parity_a
    hidden_states = np.zeros((4, 16))
    hidden_states[0, 0] = 1  # 2 is even
    hidden_states[1, 0] = 1  # 4 is even
    hidden_states[2, 0] = -1 # 1 is odd
    hidden_states[3, 0] = 1  # 10 is even

    accs = train_probes(hidden_states, inputs, targets, prime)

    assert "parity_a" in accs
    assert "parity_b" in accs
    assert "result_bucket" in accs

    # Should easily predict parity_a
    assert accs["parity_a"] > 0.9

def test_extract_hidden_states():
    model = ModularArithmeticTransformer(prime=59)
    inputs = torch.randint(0, 59, (100, 2))
    targets = (inputs[:, 0] + inputs[:, 1]) % 59

    h, x, y = extract_dataset_hidden_states(model, inputs, targets, torch.device("cpu"), batch_size=32)

    assert h.shape == (100, 128) # d_model
    assert x.shape == (100, 2)
    assert y.shape == (100,)
