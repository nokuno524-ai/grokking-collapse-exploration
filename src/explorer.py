"""
Interactive grokking explorer and parameter space visualization.

This module provides tools for generating 2D/3D parameter space maps,
interactive exploration using matplotlib sliders, and automated critical
point detection (grokking point, final accuracy, collapse point).
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def detect_critical_points(
    history: List[Dict[str, Any]], grokking_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Automated critical point detection.

    Args:
        history: List of dictionaries containing training metrics per step.
                 Expected keys: 'step', 'train_acc', 'test_acc'
        grokking_threshold: The accuracy threshold to consider a model 'grokked'.

    Returns:
        A dictionary containing:
            - 'grokking_point': The step where test_acc first crosses the threshold, or None.
            - 'final_accuracy': The final test accuracy.
            - 'collapse_point': The step where train_acc stops improving significantly and stays low,
                                or None if it doesn't collapse. We define collapse here as
                                failing to reach 90% train accuracy by the end.
    """
    if not history:
        return {"grokking_point": None, "final_accuracy": 0.0, "collapse_point": None}

    grokking_point = None
    collapse_point = None

    for entry in history:
        if grokking_point is None and entry.get("test_acc", 0) >= grokking_threshold:
            grokking_point = entry["step"]

    final_entry = history[-1]
    final_train_acc = final_entry.get("train_acc", 0)
    final_test_acc = final_entry.get("test_acc", 0)

    if final_train_acc < 0.9:
        # If it never memorized, consider it collapsed. We define the collapse point as the max step.
        collapse_point = final_entry["step"]

    return {
        "grokking_point": grokking_point,
        "final_accuracy": final_test_acc,
        "collapse_point": collapse_point,
    }


def plot_2d_parameter_space(
    results: List[Dict[str, Any]],
    x_param: str = "lr",
    y_param: str = "weight_decay",
    z_param: str = "final_test_acc",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Generate a 2D parameter space map.

    Args:
        results: List of result dictionaries containing parameters and metrics.
        x_param: Parameter for x-axis.
        y_param: Parameter for y-axis.
        z_param: Metric for color mapping.
        ax: Optional matplotlib axis.

    Returns:
        The matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    xs = [r[x_param] for r in results]
    ys = [r[y_param] for r in results]
    zs = [r[z_param] for r in results]

    # Create a scatter plot representing the phase diagram
    sc = ax.scatter(xs, ys, c=zs, cmap="viridis", marker="s", s=100)
    plt.colorbar(sc, ax=ax, label=z_param)

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{z_param} over {x_param} and {y_param}")

    return ax


def plot_3d_parameter_space(
    results: List[Dict[str, Any]],
    x_param: str = "lr",
    y_param: str = "weight_decay",
    z_param: str = "collapse_level",
    c_param: str = "final_test_acc",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Generate a 3D parameter space map.

    Args:
        results: List of result dictionaries containing parameters and metrics.
        x_param: Parameter for x-axis.
        y_param: Parameter for y-axis.
        z_param: Parameter for z-axis.
        c_param: Metric for color mapping.
        ax: Optional 3D matplotlib axis.

    Returns:
        The matplotlib axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    xs = [r[x_param] for r in results]
    ys = [r[y_param] for r in results]
    zs = [r[z_param] for r in results]
    cs = [r[c_param] for r in results]

    sc = ax.scatter(xs, ys, zs, c=cs, cmap="viridis", s=50)  # type: ignore
    plt.colorbar(sc, ax=ax, label=c_param)

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(z_param)  # type: ignore
    ax.set_title(f"3D Phase Diagram of {c_param}")

    return ax


def interactive_explorer(results: List[Dict[str, Any]]) -> None:
    """
    Interactive exploration using matplotlib sliders.

    Shows a 2D plot of lr vs weight_decay, with a slider for collapse_level.
    """
    # Extract unique collapse levels
    collapse_levels = sorted(list(set(r.get("collapse_level", 0.0) for r in results)))
    if not collapse_levels:
        print("No collapse levels found in results.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    def get_data_for_cl(cl: float) -> Tuple[List[float], List[float], List[float]]:
        subset = [r for r in results if np.isclose(r.get("collapse_level", 0.0), cl)]
        xs = [r.get("lr", 0.0) for r in subset]
        ys = [r.get("weight_decay", 0.0) for r in subset]
        zs = [r.get("final_test_acc", 0.0) for r in subset]
        return xs, ys, zs

    init_cl = collapse_levels[0]
    xs, ys, zs = get_data_for_cl(init_cl)
    sc = ax.scatter(xs, ys, c=zs, cmap="viridis", vmin=0, vmax=1.0, marker="s", s=100)
    plt.colorbar(sc, ax=ax, label="final_test_acc")

    ax.set_xlabel("lr")
    ax.set_ylabel("weight_decay")
    ax.set_title(f"Final Test Acc at collapse_level={init_cl:.2f}")

    axcolor = "lightgoldenrodyellow"
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)  # type: ignore

    # We use a slider over the indices of the collapse levels
    slider = Slider(
        ax_slider,
        "Collapse Level Index",
        0,
        len(collapse_levels) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val: float) -> None:
        idx = int(slider.val)
        cl = collapse_levels[idx]
        xs, ys, zs = get_data_for_cl(cl)
        ax.clear()

        # Re-plot
        if xs:
            ax.scatter(
                xs, ys, c=zs, cmap="viridis", vmin=0, vmax=1.0, marker="s", s=100
            )
        ax.set_xlabel("lr")
        ax.set_ylabel("weight_decay")
        ax.set_title(f"Final Test Acc at collapse_level={cl:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
