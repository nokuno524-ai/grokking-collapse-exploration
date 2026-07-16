def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

old_test = """def test_track_gradient_norms():
    m1 = ModularArithmeticTransformer()
    m2 = ModularArithmeticTransformer()

    for param in m2.parameters():
        param.data += 0.1

    ckpt1 = {'model_state': m1.state_dict()}
    ckpt2 = {'model_state': m2.state_dict()}

    norms = track_gradient_norms([ckpt1, ckpt2])
    assert len(norms) == 1
    assert norms[0] > 0"""

new_test = """def test_track_gradient_norms():
    m1 = ModularArithmeticTransformer()
    m2 = ModularArithmeticTransformer()

    for param in m2.parameters():
        param.data += 0.1

    ckpt1 = {'model_state': m1.state_dict()}
    ckpt2 = {'model_state': m2.state_dict()}

    norms_dict = track_gradient_norms([ckpt1, ckpt2])
    assert len(norms_dict) > 0
    for name, norms in norms_dict.items():
        assert len(norms) == 1
        assert norms[0] > 0"""

replace_in_file("tests/test_gradient_flow.py", old_test, new_test)
