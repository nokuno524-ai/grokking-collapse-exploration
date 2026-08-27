import re

with open("tests/test_transplant.py", "r") as f:
    content = f.read()

# Add a test for in_proj_bias
bias_test_addition = """
    # Check out_proj
    out_proj_base = base_sd["transformer.layers.1.self_attn.out_proj.weight"]
    out_proj_donor = donor_sd["transformer.layers.1.self_attn.out_proj.weight"]
    out_proj_patched = patched_sd["transformer.layers.1.self_attn.out_proj.weight"]

    # For out_proj, head 0 inputs are at columns [0:8]
    assert torch.all(out_proj_patched[:, 0:8] == out_proj_donor[:, 0:8])
    assert torch.all(out_proj_patched[:, 8:16] == out_proj_base[:, 8:16])

    # Check in_proj_bias
    in_proj_bias_base = base_sd["transformer.layers.1.self_attn.in_proj_bias"]
    in_proj_bias_donor = donor_sd["transformer.layers.1.self_attn.in_proj_bias"]
    in_proj_bias_patched = patched_sd["transformer.layers.1.self_attn.in_proj_bias"]

    # Q bias
    assert torch.all(in_proj_bias_patched[0:8] == in_proj_bias_donor[0:8])
    assert torch.all(in_proj_bias_patched[8:16] == in_proj_bias_base[8:16])
    # K bias
    assert torch.all(in_proj_bias_patched[16:24] == in_proj_bias_donor[16:24])
    assert torch.all(in_proj_bias_patched[24:32] == in_proj_bias_base[24:32])
    # V bias
    assert torch.all(in_proj_bias_patched[32:40] == in_proj_bias_donor[32:40])
    assert torch.all(in_proj_bias_patched[40:48] == in_proj_bias_base[40:48])
"""

content = content.replace("""    # Check out_proj
    out_proj_base = base_sd["transformer.layers.1.self_attn.out_proj.weight"]
    out_proj_donor = donor_sd["transformer.layers.1.self_attn.out_proj.weight"]
    out_proj_patched = patched_sd["transformer.layers.1.self_attn.out_proj.weight"]

    # For out_proj, head 0 inputs are at columns [0:8]
    assert torch.all(out_proj_patched[:, 0:8] == out_proj_donor[:, 0:8])
    assert torch.all(out_proj_patched[:, 8:16] == out_proj_base[:, 8:16])""", bias_test_addition)

with open("tests/test_transplant.py", "w") as f:
    f.write(content)
