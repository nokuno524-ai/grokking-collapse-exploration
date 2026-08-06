import torch
import copy
from src.model import ModularArithmeticTransformer
from src.analysis.interventions import ablate_head, counterfactual_patch, run_intervention_suite

def test_ablate_head():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    model = ablate_head(model, 1)

    out_proj_weight = model.transformer.layers[0].self_attn.out_proj.weight
    head_dim = 128 // 4
    start_idx = 1 * head_dim
    end_idx = 2 * head_dim

    # Check that the weights for head 1 are exactly zero
    assert torch.all(out_proj_weight[:, start_idx:end_idx] == 0)
    # Check that the weights for head 0 are not zero
    assert not torch.all(out_proj_weight[:, 0:start_idx] == 0)

def test_counterfactual_patch():
    model_a = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    model_b = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)

    model_b = counterfactual_patch(model_a, model_b, layer=0)

    x = torch.randint(0, 59, (4, 2))
    # Trigger hooks
    _ = model_a(x)
    out_b = model_b(x)
    assert out_b.shape == (4, 59)

def test_run_intervention_suite():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    dataset = torch.utils.data.TensorDataset(torch.randint(0, 59, (32, 2)), torch.randint(0, 59, (32,)))

    results = run_intervention_suite(model, dataset)
    assert len(results) == 4
    for i in range(4):
        assert f'head_{i}_loss_diff' in results
        assert isinstance(results[f'head_{i}_loss_diff'], float)


def test_ablate_mlp_neurons():
    from src.analysis.interventions import ablate_mlp_neurons
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    model = ablate_mlp_neurons(model, layer_idx=0, neuron_indices=[5, 10, 42])

    linear2_weight = model.transformer.layers[0].linear2.weight
    assert torch.all(linear2_weight[:, 5] == 0)
    assert torch.all(linear2_weight[:, 10] == 0)
    assert torch.all(linear2_weight[:, 42] == 0)
    assert not torch.all(linear2_weight[:, 0] == 0)
