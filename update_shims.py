import re

# Update src/analysis.py to use grokkit
with open("src/analysis.py", "r") as f:
    text = f.read()

text = text.replace("def _ordered_condition_dirs(results_dir: Path) -> List[Path]:",
                    "from grokkit.parser import collect_runs\nfrom grokkit.figures import plot_training_trajectory as _pt, plot_grokking_comparison as _pg\ndef _ordered_condition_dirs(results_dir: Path) -> List[Path]:")
text = text.replace("def plot_training_trajectory(results_dir: Path, output_path: Optional[Path] = None):",
                    "def plot_training_trajectory(results_dir: Path, output_path: Optional[Path] = None):\n    runs = collect_runs(results_dir)\n    if runs:\n        _pt(runs, output_path or results_dir / 'training_trajectories.png')\n    return\n")
text = text.replace("def plot_grokking_comparison(results_dir: Path, output_path: Optional[Path] = None):",
                    "def plot_grokking_comparison(results_dir: Path, output_path: Optional[Path] = None):\n    runs = collect_runs(results_dir)\n    if runs:\n        _pg(runs, output_path or results_dir / 'grokking_comparison.png')\n    return\n")

with open("src/analysis.py", "w") as f:
    f.write(text)

# Update analysis/exp_c_grid_analysis.py
with open("analysis/exp_c_grid_analysis.py", "r") as f:
    text = f.read()

text = text.replace("def find_fourier_cliff(rows, wds, noises, threshold=0.20):",
                    "from grokkit.cliff import find_cliff as _fc\ndef find_fourier_cliff(rows, wds, noises, threshold=0.20):\n    return _fc(rows, 'wd', 'noise', 'final_fourier_concentration', threshold, 'below')\n")
text = text.replace("def collect_runs():",
                    "from grokkit.parser import collect_runs as _cr\ndef collect_runs():\n    return _cr(GRID_ROOT)\n")

with open("analysis/exp_c_grid_analysis.py", "w") as f:
    f.write(text)

# Update analysis/grid_analysis.py
with open("analysis/grid_analysis.py", "r") as f:
    text = f.read()

text = text.replace("def collect_runs():",
                    "from grokkit.parser import collect_runs as _cr\ndef collect_runs():\n    return _cr(GRID_ROOT)\n")

with open("analysis/grid_analysis.py", "w") as f:
    f.write(text)
