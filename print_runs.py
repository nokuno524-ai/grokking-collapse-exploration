import json
from src.grokkit.parser import collect_runs
from pathlib import Path

runs = collect_runs(Path("results/grid"))
if runs:
    print(runs[0].keys())
