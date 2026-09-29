import torch
from src.model import ModularArithmeticTransformer
from src.transplant.patching import patch_state_dict, get_keys_for_component

def test_patch_state_dict():
    model_base = ModularArithmeticTransformer(d_model=32, d_ff=128, n_heads=2)
    model_donor = ModularArithmeticTransformer(d_model=32, d_ff=128, n_heads=2)

    sd_base = model_base.state_dict()
    sd_donor = model_donor.state_dict()

    components_to_patch = ["linear1"]
    patched_sd = patch_state_dict(sd_base, sd_donor, components_to_patch)

    assert torch.allclose(patched_sd["transformer.layers.0.linear1.weight"], sd_donor["transformer.layers.0.linear1.weight"])
    assert torch.allclose(patched_sd["token_embed.weight"], sd_base["token_embed.weight"])

def test_patch_with_key_mapping():
    model_base = ModularArithmeticTransformer(d_model=32, d_ff=128, n_heads=2)
    sd_base = model_base.state_dict()

    donor_sd = {
        "donor_linear1_weight": torch.randn_like(sd_base["transformer.layers.0.linear1.weight"])
    }

    key_mapping = {
        "transformer.layers.0.linear1.weight": "donor_linear1_weight"
    }

    # We patch only the exact mapped key.
    # Because get_keys_for_component gets both weight and bias, and bias is missing in donor_sd,
    # let's modify sd_base to not have bias for this test or map it.
    del sd_base["transformer.layers.0.linear1.bias"]

    patched_sd = patch_state_dict(sd_base, donor_sd, ["linear1"], key_mapping=key_mapping)

    assert torch.allclose(patched_sd["transformer.layers.0.linear1.weight"], donor_sd["donor_linear1_weight"])
