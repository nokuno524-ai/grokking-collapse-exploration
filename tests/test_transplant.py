import pytest
import torch
from typing import Dict
from src.transplant_rescue import patch_state_dict

def test_patch_state_dict_transplant():
    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(384, 128),
        "other_param": torch.ones(10)
    }
    donor_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(384, 128),
        "other_param": torch.zeros(10)
    }

    # We patch self_attn_in_proj
    patched = patch_state_dict(base_sd, donor_sd, "self_attn_in_proj")

    # Check that it took the donor's value for the matched component
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"] == 0.0)

    # Check that it kept the base's value for the unmatched component
    assert torch.all(patched["other_param"] == 1.0)

def test_patch_state_dict_shape_mismatch():
    base_sd = {
        "output_head.weight": torch.ones(59, 128)
    }
    donor_sd = {
        "output_head.weight": torch.ones(60, 128) # Mismatched shape
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(base_sd, donor_sd, "output_head")
