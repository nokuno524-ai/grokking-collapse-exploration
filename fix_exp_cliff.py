with open("analysis/exp_c_grid_analysis.py", "r") as f:
    content = f.read()

import_str = """
from grokkit.cliff import find_cliff as _fc
def find_fourier_cliff(rows, wds, noises, threshold=0.20):
    return _fc(rows, "wd", "noise", "final_fourier_concentration", threshold, "below")
"""

content = content.replace("def write_csv(", import_str + "\ndef write_csv(")

with open("analysis/exp_c_grid_analysis.py", "w") as f:
    f.write(content)
