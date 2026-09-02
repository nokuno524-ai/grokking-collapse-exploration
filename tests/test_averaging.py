import torch
import pytest
from src.averaging.weight_averaging import check_keys_match, interpolate_weights, average_weights

def test_check_keys_match():
    sd1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
    sd2 = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.1])}

    # Should not raise
    check_keys_match(sd1, sd2)

    # Key mismatch
    sd3 = {"w": torch.tensor([3.0, 4.0])}
    with pytest.raises(ValueError, match="State dict keys do not match"):
        check_keys_match(sd1, sd3)

    # Shape mismatch
    sd4 = {"w": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([0.5])}
    with pytest.raises(ValueError, match="Shape mismatch"):
        check_keys_match(sd1, sd4)

def test_interpolate_weights():
    sd_pre = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([10.0])}
    sd_post = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([20.0])}

    # alpha = 1.0 -> pre
    res = interpolate_weights(sd_pre, sd_post, 1.0)
    assert torch.allclose(res["w"], sd_pre["w"])
    assert torch.allclose(res["b"], sd_pre["b"])

    # alpha = 0.0 -> post
    res = interpolate_weights(sd_pre, sd_post, 0.0)
    assert torch.allclose(res["w"], sd_post["w"])
    assert torch.allclose(res["b"], sd_post["b"])

    # alpha = 0.5 -> mean
    res = interpolate_weights(sd_pre, sd_post, 0.5)
    assert torch.allclose(res["w"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(res["b"], torch.tensor([15.0]))

    # Check int tensors (like num_batches_tracked)
    sd_pre_int = {"n": torch.tensor([1], dtype=torch.int64)}
    sd_post_int = {"n": torch.tensor([3], dtype=torch.int64)}
    res = interpolate_weights(sd_pre_int, sd_post_int, 0.5)
    assert res["n"].dtype == torch.int64
    assert res["n"].item() == 2

def test_average_weights():
    sd1 = {"w": torch.tensor([1.0, 2.0])}
    sd2 = {"w": torch.tensor([3.0, 4.0])}
    sd3 = {"w": torch.tensor([5.0, 6.0])}

    # 2 dicts
    res = average_weights([sd1, sd2])
    assert torch.allclose(res["w"], torch.tensor([2.0, 3.0]))

    # 3 dicts
    res = average_weights([sd1, sd2, sd3])
    assert torch.allclose(res["w"], torch.tensor([3.0, 4.0]))

    # 1 dict
    res = average_weights([sd1])
    assert torch.allclose(res["w"], sd1["w"])

    # empty list
    with pytest.raises(ValueError, match="List of state dicts is empty"):
        average_weights([])
