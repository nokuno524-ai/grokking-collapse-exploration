"""Streamlit dashboard for grokking and model collapse visualization."""

import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import streamlit as st

# Add src to sys.path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from explorer import plot_2d_parameter_space, plot_3d_parameter_space  # type: ignore
except ImportError:
    pass


def generate_mock_data() -> List[Dict[str, Any]]:
    """Generate mock data for visualization purposes."""
    import itertools
    import random

    # Simple mock data generation for demonstration
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    wds = [0.1, 0.5, 1.0, 2.0]
    cls = [0.0, 0.2, 0.4, 0.6]

    results = []
    for lr, wd, cl in itertools.product(lrs, wds, cls):
        # A pseudo-function to simulate final test accuracy
        # grokking prefers specific lr/wd combinations and low collapse
        acc = 0.95 * (1.0 - cl) * min(1.0, 1e-3 / lr) * min(1.0, wd / 1.0)
        acc = max(0.0, min(1.0, acc + random.uniform(-0.1, 0.1)))

        results.append(
            {"lr": lr, "weight_decay": wd, "collapse_level": cl, "final_test_acc": acc}
        )
    return results


st.title("Grokking & Model Collapse Explorer")

# Sidebar
st.sidebar.header("Data Selection")
data_source = st.sidebar.selectbox(
    "Data Source", ["Mock Data", "Results JSON (Not Implemented)"]
)

if data_source == "Mock Data":
    results = generate_mock_data()
else:
    results = []

st.sidebar.header("Visualization Settings")
viz_type = st.sidebar.selectbox(
    "Plot Type", ["2D Parameter Space", "3D Parameter Space"]
)

st.sidebar.subheader("Plot Parameters")
if viz_type == "2D Parameter Space":
    collapse_levels = sorted(list(set(r["collapse_level"] for r in results)))
    selected_cl = st.sidebar.selectbox("Collapse Level", collapse_levels)

    st.subheader(f"2D Phase Diagram at Collapse Level {selected_cl}")

    # Filter data
    subset = [r for r in results if r["collapse_level"] == selected_cl]

    if subset:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_2d_parameter_space(
            subset,
            x_param="lr",
            y_param="weight_decay",
            z_param="final_test_acc",
            ax=ax,
        )
        st.pyplot(fig)
    else:
        st.write("No data available for the selected collapse level.")

elif viz_type == "3D Parameter Space":
    st.subheader("3D Phase Diagram")

    if results:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_parameter_space(
            results,
            x_param="lr",
            y_param="weight_decay",
            z_param="collapse_level",
            c_param="final_test_acc",
            ax=ax,
        )
        st.pyplot(fig)
    else:
        st.write("No data available.")

st.sidebar.markdown("---")
st.sidebar.info(
    "This interactive explorer lets you visualize how learning rate, weight decay, "
    "and synthetic data contamination (collapse) affect grokking."
)
