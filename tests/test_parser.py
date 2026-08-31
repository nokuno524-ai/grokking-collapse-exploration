import json
import math
from pathlib import Path
from grokkit.parser import parse_run_results, collect_runs

def test_parse_run_results_nans_and_dupes(tmp_path: Path):
    res_file = tmp_path / "results.json"
    data = {
        "history": [
            {"step": 100, "val": 0.5},
            {"step": 100, "val": 0.6}, # duplicate step
            {"step": 200, "val": "NaN"},
            {"step": 300, "val": float('nan')}
        ]
    }
    with open(res_file, "w") as f:
        json.dump(data, f)

    parsed = parse_run_results(res_file)
    history = parsed["history"]

    assert len(history) == 3
    assert history[0]["step"] == 100
    assert history[0]["val"] == 0.5

    assert history[1]["step"] == 200
    assert math.isnan(history[1]["val"])

    assert history[2]["step"] == 300
    assert math.isnan(history[2]["val"])

def test_collect_runs(tmp_path: Path):
    d1 = tmp_path / "cond1"
    d1.mkdir()
    with open(d1 / "results.json", "w") as f:
        json.dump({"final_test_acc": 0.99}, f)

    d2 = tmp_path / "cond2"
    d2.mkdir()
    with open(d2 / "results.json", "w") as f:
        json.dump({"final_test_acc": 0.50}, f)

    runs = collect_runs(tmp_path)
    assert len(runs) == 2
    assert any(r["condition"] == "cond1" for r in runs)
    assert any(r["condition"] == "cond2" for r in runs)
