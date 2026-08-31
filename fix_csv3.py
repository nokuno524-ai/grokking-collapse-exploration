for filepath in ["analysis/grid_analysis.py", "analysis/exp_c_grid_analysis.py"]:
    with open(filepath, "r") as f:
        content = f.read()

    # Actually wait, let's see how write_csv is defined.
    # We can just manually filter keys before passing to write_csv!
