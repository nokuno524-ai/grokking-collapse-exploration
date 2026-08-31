for filepath in ["analysis/grid_analysis.py", "analysis/exp_c_grid_analysis.py"]:
    with open(filepath, "r") as f:
        content = f.read()

    # The actual string is `csv.DictWriter(f, fieldnames=cols)`
    content = content.replace("csv.DictWriter(f, fieldnames=cols)", "csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')")

    with open(filepath, "w") as f:
        f.write(content)
