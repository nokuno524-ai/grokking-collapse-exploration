import re

for filepath in ["analysis/grid_analysis.py", "analysis/exp_c_grid_analysis.py"]:
    with open(filepath, "r") as f:
        content = f.read()

    # The original write_csv just takes dicts. Let's make it ignore extra fields (extrasaction='ignore')
    content = content.replace("csv.DictWriter(f, fieldnames=fields)", "csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')")

    with open(filepath, "w") as f:
        f.write(content)
