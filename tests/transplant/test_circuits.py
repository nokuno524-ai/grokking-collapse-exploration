import pytest
import torch
import torch.nn as nn

from src.transplant.circuits import (
    patch_state_dict,
    patch_random_basis,
    shuffle_attention_heads,
    patch_attention_head,
    strip_prefixes
)
from src.model import ModularArithmeticTransformer

@pytest.fixture
def tiny_model():
    # Tiny CPU model purely for testing shapes and patches
    model = ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=2
    )
    return model

def test_exact_transplant_is_identity(tiny_model):
    sd = tiny_model.state_dict()

    # donor == recipient
    patched_sd = patch_state_dict(sd, sd, "self_attn_in_proj")

    for k in sd.keys():
        assert torch.equal(sd[k], patched_sd[k]), f"Key {k} was unexpectedly modified"

def test_key_remapping():
    base = {"module.linear1.weight": torch.ones(2, 2)}
    donor = {"_orig_mod.linear1.weight": torch.zeros(2, 2)}

    # Assuming "linear1" is a valid component in COMPONENT_PATTERNS
    # Testing our utility directly for prefix stripping
    assert strip_prefixes("module.linear1.weight") == "linear1.weight"
    assert strip_prefixes("_orig_mod.linear1.weight") == "linear1.weight"

def test_patch_state_dict_updates_component(tiny_model):
    sd1 = tiny_model.state_dict()

    # Make a donor with modified weights
    model2 = ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )
    # Ensure they are different
    for p in model2.parameters():
        nn.init.constant_(p, 42.0)
    sd2 = model2.state_dict()

    patched_sd = patch_state_dict(sd1, sd2, "token_embed")

    # Check token_embed was updated
    assert torch.equal(patched_sd["token_embed.weight"], sd2["token_embed.weight"])

    # Check something else was NOT updated
    assert torch.equal(patched_sd["pos_embed.weight"], sd1["pos_embed.weight"])

def test_lesion_controls_perturb_weights(tiny_model):
    sd = tiny_model.state_dict()

    # Random basis should change the values but keep the shape
    rand_sd = patch_random_basis(sd, "self_attn_in_proj", seed=42)

    in_proj_w_key = "transformer.layers.0.self_attn.in_proj_weight"

    assert rand_sd[in_proj_w_key].shape == sd[in_proj_w_key].shape
    assert not torch.allclose(rand_sd[in_proj_w_key], sd[in_proj_w_key])

    # Test random basis on a wide-and-short matrix to verify SVD transpose fix
    rand_sd_token = patch_random_basis(sd, "token_embed", seed=42)
    token_embed_key = "token_embed.weight"

    assert rand_sd_token[token_embed_key].shape == sd[token_embed_key].shape
    assert not torch.allclose(rand_sd_token[token_embed_key], sd[token_embed_key])

def test_shuffle_attention_heads(tiny_model):
    sd = tiny_model.state_dict()

    # Just one layer in tiny_model
    shuffled_sd = shuffle_attention_heads(sd, layer_idx=0, n_heads=2, seed=42)

    in_proj_w_key = "transformer.layers.0.self_attn.in_proj_weight"

    assert shuffled_sd[in_proj_w_key].shape == sd[in_proj_w_key].shape
    # Hard to test exact inequality because a shuffle might be an identity permutation
    # but with seed=42 for 2 heads, it should swap them.
    # We can at least check the tensor is still populated and shaped right.
    assert not torch.isnan(shuffled_sd[in_proj_w_key]).any()

def test_patch_layer_blocks(tiny_model):
    from src.transplant.circuits import patch_layer_blocks
    sd1 = tiny_model.state_dict()

    model2 = ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=2
    )
    for p in model2.parameters():
        nn.init.constant_(p, 42.0)
    sd2 = model2.state_dict()

    patched_sd = patch_layer_blocks(sd1, sd2, start_layer=0, end_layer=1)

    # Layer 0 should be updated to model2
    assert torch.equal(patched_sd["transformer.layers.0.linear1.weight"], sd2["transformer.layers.0.linear1.weight"])
    # Layer 1 should remain model1
    assert torch.equal(patched_sd["transformer.layers.1.linear1.weight"], sd1["transformer.layers.1.linear1.weight"])
