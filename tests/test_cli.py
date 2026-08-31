import subprocess
import json
from pathlib import Path

def test_cli_analyze(tmp_path: Path):
    d1 = tmp_path / "cond1"
    d1.mkdir()
    with open(d1 / "results.json", "w") as f:
        json.dump({
            "final_test_acc": 0.99,
            "final_fourier_concentration": 0.15,
            "grokked": True,
            "grokking_step": 1000
        }, f)

    res = subprocess.run(["grokkit", "analyze", str(tmp_path), "--json"], capture_output=True, text=True)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert "cond1" in data
    assert data["cond1"]["test_acc_mean"] == 0.99

def test_cli_compare(tmp_path: Path):
    d1 = tmp_path / "dir1" / "condA"
    d1.mkdir(parents=True)
    with open(d1 / "results.json", "w") as f:
        json.dump({"final_test_acc": 0.9}, f)

    d2 = tmp_path / "dir2" / "condB"
    d2.mkdir(parents=True)
    with open(d2 / "results.json", "w") as f:
        json.dump({"final_test_acc": 0.5}, f)

    res = subprocess.run(["grokkit", "compare", str(tmp_path / "dir1"), str(tmp_path / "dir2"), "--json"], capture_output=True, text=True)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert "condA" in data
    assert "condB" in data

def test_cli_cliff(tmp_path: Path):
    d1 = tmp_path / "run1"
    d1.mkdir()
    with open(d1 / "results.json", "w") as f:
        json.dump({
            "config": {"wd": 0.1, "noise": 0.0},
            "final_fourier_concentration": 0.25
        }, f)

    d2 = tmp_path / "run2"
    d2.mkdir()
    with open(d2 / "results.json", "w") as f:
        json.dump({
            "config": {"wd": 0.1, "noise": 0.1},
            "final_fourier_concentration": 0.15
        }, f)

    res = subprocess.run(["grokkit", "cliff", str(tmp_path), "--json"], capture_output=True, text=True)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert "0.1" in data
    assert data["0.1"] == 0.1
