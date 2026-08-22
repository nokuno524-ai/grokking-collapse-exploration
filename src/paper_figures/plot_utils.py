import matplotlib.pyplot as plt
import matplotlib as mpl

def set_style():
    """Sets a consistent publication-ready style for matplotlib figures."""
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['figure.titlesize'] = 12
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = '--'
    # Use vector-friendly output
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

def get_color_palette():
    """Returns a consistent color palette for conditions."""
    return {
        "pure": "#2ecc71",          # Green
        "low_collapse": "#3498db",  # Blue
        "medium_collapse": "#f39c12", # Orange
        "high_collapse": "#e74c3c", # Red
        "severe_collapse": "#8e44ad" # Purple
    }

def condition_to_color(noise_fraction, collapse_severity=None):
    """Map noise fraction to color for consistent plotting."""
    if noise_fraction == 0.0:
        return "#2ecc71" # pure
    elif noise_fraction <= 0.05:
        return "#3498db"
    elif noise_fraction <= 0.15:
        return "#f39c12"
    elif noise_fraction <= 0.30:
        return "#e74c3c"
    else:
        return "#8e44ad"
