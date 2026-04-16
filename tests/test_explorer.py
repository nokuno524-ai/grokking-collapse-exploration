import matplotlib.pyplot as plt

from src.explorer import (detect_critical_points, plot_2d_parameter_space,
                          plot_3d_parameter_space)


def test_detect_critical_points():
    # Test grokking detection
    history_grok = [
        {"step": 100, "train_acc": 0.5, "test_acc": 0.5},
        {"step": 200, "train_acc": 0.95, "test_acc": 0.6},
        {"step": 300, "train_acc": 0.99, "test_acc": 0.96},  # Grokking point
        {"step": 400, "train_acc": 1.0, "test_acc": 0.98},
    ]

    res = detect_critical_points(history_grok)
    assert res["grokking_point"] == 300
    assert res["final_accuracy"] == 0.98
    assert res["collapse_point"] is None

    # Test collapse detection
    history_collapse = [
        {"step": 100, "train_acc": 0.5, "test_acc": 0.5},
        {"step": 200, "train_acc": 0.6, "test_acc": 0.55},
        {"step": 300, "train_acc": 0.65, "test_acc": 0.55},
        {
            "step": 400,
            "train_acc": 0.65,
            "test_acc": 0.56,
        },  # Collapse because final < 0.9
    ]

    res2 = detect_critical_points(history_collapse)
    assert res2["grokking_point"] is None
    assert res2["final_accuracy"] == 0.56
    assert res2["collapse_point"] == 400

    # Test empty history
    res3 = detect_critical_points([])
    assert res3["grokking_point"] is None
    assert res3["final_accuracy"] == 0.0
    assert res3["collapse_point"] is None


def test_visualization_outputs():
    # Mock data
    results = [
        {
            "lr": 1e-3,
            "weight_decay": 1.0,
            "collapse_level": 0.0,
            "final_test_acc": 0.95,
        },
        {
            "lr": 5e-3,
            "weight_decay": 0.1,
            "collapse_level": 0.2,
            "final_test_acc": 0.60,
        },
    ]

    # Test 2D plot runs without error
    ax_2d = plot_2d_parameter_space(results)
    assert isinstance(ax_2d, plt.Axes)
    plt.close("all")

    # Test 3D plot runs without error
    ax_3d = plot_3d_parameter_space(results)
    assert isinstance(ax_3d, plt.Axes)
    plt.close("all")
