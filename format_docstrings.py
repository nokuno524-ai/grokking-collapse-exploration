import re

files_to_docstring = ["src/sweep/run_dilution.py", "src/sweep/run_curriculum.py", "src/sweep/run_scaling.py", "src/analysis/aggregator.py"]

for fpath in files_to_docstring:
    with open(fpath, "r") as f:
        content = f.read()

    # Just checking they have docstrings. We added them during file creation.
    # We will just verify they exist.
    if '"""' not in content:
        print(f"Missing docstrings in {fpath}")
