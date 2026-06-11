import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.model import ModularArithmeticTransformer
from src.circuit_discovery import (
    activation_patching,
    trace_information_flow,
    find_minimal_grokking_circuit
)

@pytest.fixture
def mock_model():
    return ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1
    )

def test_activation_patching(mock_model):
    clean = torch.tensor([[1, 2]])
    corrupted = torch.tensor([[3, 4]])

    def metric_fn(logits):
        return logits[0, 3].item()  # Just return logit for class 3

    # Patch embedding
    embed_val = activation_patching(mock_model, clean, corrupted, "embed", metric_fn)
    assert isinstance(embed_val, float)

    # Patch attention
    attn_val = activation_patching(mock_model, clean, corrupted, "attn", metric_fn)
    assert isinstance(attn_val, float)

    # Patch FFN
    ffn_val = activation_patching(mock_model, clean, corrupted, "ffn", metric_fn)
    assert isinstance(ffn_val, float)

def test_trace_information_flow(mock_model):
    inputs = torch.tensor([[1, 2], [3, 4]])
    targets = torch.tensor([3, 2])

    effects = trace_information_flow(mock_model, inputs, targets)

    assert "baseline_clean" in effects
    assert "baseline_corrupted" in effects
    assert "embed" in effects
    assert "attn" in effects
    assert "ffn" in effects

def test_find_minimal_grokking_circuit(mock_model):
    inputs = torch.tensor([[1, 2], [3, 4], [0, 1], [2, 2]])
    targets = torch.tensor([3, 2, 1, 4])
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    # Run with a very low threshold so it doesn't just return all heads
    # Note: untrained model will likely have low accuracy
    circuit = find_minimal_grokking_circuit(mock_model, dataloader, device=torch.device("cpu"), acc_threshold=-1.0)

    assert isinstance(circuit, list)
    assert len(circuit) > 0
    assert all(isinstance(h, int) for h in circuit)
