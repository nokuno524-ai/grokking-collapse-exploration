import pytest
import torch
import torch.nn as nn
from copy import deepcopy

from src.model import ModularArithmeticTransformer
from src.transplant.sweep import graft_context

def test_transplant_graft_restore_bit_identical():
    """Test that applying a graft and exiting the context restores host weights exactly."""
    host = ModularArithmeticTransformer(59, 128, 4, 512, 1)
    donor = ModularArithmeticTransformer(59, 128, 4, 512, 1)

    # Randomize donor
    for p in donor.parameters():
        nn.init.normal_(p)

    original_host_sd = deepcopy(host.state_dict())

    with graft_context(host, donor, ["self_attn_in_proj"], "donor"):
        # Check that they differ
        differ = False
        for k, v in host.state_dict().items():
            if "self_attn" in k and not torch.equal(v, original_host_sd[k]):
                differ = True
        assert differ, "Graft failed to modify host weights during context."

    # Check that they are bit-identical after
    for k, v in host.state_dict().items():
        assert torch.equal(v, original_host_sd[k]), f"Weight {k} was not restored."


def test_transplant_sham_bit_identical():
    """Test that sham graft leaves the model bit-identical during the context."""
    host = ModularArithmeticTransformer(59, 128, 4, 512, 1)
    donor = ModularArithmeticTransformer(59, 128, 4, 512, 1)

    original_host_sd = deepcopy(host.state_dict())

    with graft_context(host, donor, ["self_attn_in_proj"], "sham"):
        for k, v in host.state_dict().items():
            assert torch.equal(v, original_host_sd[k]), f"Sham graft modified weight {k}."


def test_transplant_changes_activations():
    """Test that grafting donor weights actually alters model behavior."""
    host = ModularArithmeticTransformer(59, 128, 4, 512, 1)
    donor = ModularArithmeticTransformer(59, 128, 4, 512, 1)

    # Initialize differently
    torch.manual_seed(10)
    for p in host.parameters():
        nn.init.normal_(p)

    torch.manual_seed(20)
    for p in donor.parameters():
        nn.init.normal_(p)

    x = torch.randint(0, 59, (4, 2))

    host.eval()
    donor.eval()

    with torch.no_grad():
        host_out_orig = host(x)

    with graft_context(host, donor, ["self_attn_in_proj", "linear1"], "donor"):
        with torch.no_grad():
            host_out_graft = host(x)

    assert not torch.allclose(host_out_orig, host_out_graft, atol=1e-5), "Graft did not change output activations."

    # Confirm rest is still identical
    with torch.no_grad():
        host_out_restored = host(x)

    assert torch.allclose(host_out_orig, host_out_restored, atol=1e-6), "Restored host output does not match original."
