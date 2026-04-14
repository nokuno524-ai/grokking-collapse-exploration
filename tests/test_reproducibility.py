import pytest
import subprocess
import json
import os

def run_train(seed):
    output_dir = f"tests/test_reproducibility_output_{seed}"
    subprocess.run([
        ".venv/bin/python", "src/train.py",
        "--max-steps", "10",
        "--output-dir", output_dir
    ], capture_output=True)
    with open(f"{output_dir}/pure/results.json") as f:
        history = json.load(f)["history"]
    return history

def test_training_reproducibility():
    hist1 = run_train(1)
    hist2 = run_train(1)

    # Assert exact match of history dicts
    assert hist1 == hist2
