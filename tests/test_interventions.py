import pytest
import torch
from pathlib import Path
from src.model import ModularArithmeticTransformer
from src.analysis.interventions import ablate_head, ablate_mlp_neuron, score_head_importances

@pytest.fixture
def model():
    # Tiny model for testing
    return ModularArithmeticTransformer(
        prime=59, d_model=16, n_heads=2, d_ff=32, n_layers=2
    )

def test_ablate_head(model):
    layer_idx = 0
    head_idx = 0

    layer = model.transformer.layers[layer_idx]
    orig_weight = layer.self_attn.out_proj.weight.clone()

    with ablate_head(model, layer_idx, head_idx):
        ablated_weight = layer.self_attn.out_proj.weight
        head_dim = model.d_model // model.n_heads
        # First half should be zeroed out
        assert torch.all(ablated_weight[:, :head_dim] == 0)
        # Second half should remain intact
        assert torch.all(ablated_weight[:, head_dim:] == orig_weight[:, head_dim:])

    # Weights should be restored
    assert torch.all(layer.self_attn.out_proj.weight == orig_weight)

def test_ablate_mlp_neuron(model):
    layer_idx = 0
    neuron_idx = 5

    layer = model.transformer.layers[layer_idx]
    orig_l1_w = layer.linear1.weight.clone()
    orig_l2_w = layer.linear2.weight.clone()

    with ablate_mlp_neuron(model, layer_idx, neuron_idx):
        ablated_l1_w = layer.linear1.weight
        ablated_l2_w = layer.linear2.weight

        # Neuron 5 output from l1 is zeroed
        assert torch.all(ablated_l1_w[neuron_idx, :] == 0)
        # Neuron 5 input to l2 is zeroed
        assert torch.all(ablated_l2_w[:, neuron_idx] == 0)

    # Weights should be restored
    assert torch.all(layer.linear1.weight == orig_l1_w)
    assert torch.all(layer.linear2.weight == orig_l2_w)

def test_score_head_importances(tmp_path, model):
    # Dummy data loader
    inputs = torch.randint(0, 59, (10, 2))
    targets = (inputs[:, 0] + inputs[:, 1]) % 59
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    output_csv = tmp_path / "importances.csv"

    results = score_head_importances(model, loader, torch.device("cpu"), output_csv)

    # 2 layers * 2 heads = 4 results
    assert len(results) == 4
    assert output_csv.exists()

    # Read back CSV
    import csv
    with open(output_csv, 'r') as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 4
        assert "layer" in reader[0]
        assert "acc_drop" in reader[0]
