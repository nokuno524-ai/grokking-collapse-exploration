with open("analysis/exp_c_grid_analysis.py", "r") as f:
    content = f.read()

content = content.replace("csv.DictWriter(f, fieldnames=fields)", "csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')")
content = content.replace("csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')", "csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')")

with open("analysis/exp_c_grid_analysis.py", "w") as f:
    f.write(content)
