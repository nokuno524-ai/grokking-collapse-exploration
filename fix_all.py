# Restore actual original logic from analysis files but swap out just the parser and cliff finders
import sys

def patch_grid_analysis():
    with open("analysis/grid_analysis.py", "r") as f:
        content = f.read()

    # We want to replace collect_runs definition
    import re
    # Remove old collect_runs
    content = re.sub(r'def collect_runs\(\).*?return runs\n', '', content, flags=re.DOTALL)

    # Add new import
    import_str = """from grokkit.parser import collect_runs as _cr
def collect_runs():
    rows = _cr(GRID_ROOT)
    # alias keys for legacy scripts
    for r in rows:
        if "collapse_level" in r: r["level"] = r["collapse_level"]
        if "collapse_severity" in r: r["severity"] = r["collapse_severity"]
    return rows
"""
    content = content.replace('def write_csv(', import_str + '\ndef write_csv(')

    # Remove config keys before csv writing because csv.DictWriter with extrasaction='raise' throws
    # Actually wait, old collect_runs manually built a dict.
    # Let's see how old collect_runs was built:
    # return [ { "level": config["collapse_level"], "severity": config["collapse_severity"], "seed": config["seed"], ... } ]

    with open("analysis/grid_analysis.py", "w") as f:
        f.write(content)


def patch_exp_c_grid_analysis():
    with open("analysis/exp_c_grid_analysis.py", "r") as f:
        content = f.read()

    import re
    # Remove old collect_runs
    content = re.sub(r'def collect_runs\(\).*?return runs\n', '', content, flags=re.DOTALL)

    # Remove old find_fourier_cliff
    content = re.sub(r'def find_fourier_cliff.*?return cliff\n', '', content, flags=re.DOTALL)

    import_str = """from grokkit.parser import collect_runs as _cr
from grokkit.cliff import find_cliff as _fc

def collect_runs():
    rows = _cr(GRID_ROOT)
    # Filter valid
    res = []
    for r in rows:
        if r.get("condition_name") == "pure": continue # Skip pure
        if "weight_decay" in r: r["wd"] = r["weight_decay"]
        if "noise_fraction" in r: r["noise"] = r["noise_fraction"]
        res.append(r)
    return res

def find_fourier_cliff(rows, wds, noises, threshold=0.20):
    return _fc(rows, "wd", "noise", "final_fourier_concentration", threshold, "below")
"""
    content = content.replace('def rank_crossover(', import_str + '\ndef rank_crossover(')

    with open("analysis/exp_c_grid_analysis.py", "w") as f:
        f.write(content)

patch_grid_analysis()
patch_exp_c_grid_analysis()
