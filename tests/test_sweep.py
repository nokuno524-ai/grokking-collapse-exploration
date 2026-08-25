import json
import os
from pathlib import Path
import subprocess
import tempfile
import csv

def test_sweep_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Generate configs
        base_config = {
            "max_steps": 100,
            "weight_decay": 1.0,
            "collapse_level": 0.0
        }

        spec = {
            "name": "test_sweep",
            "seeds": [42, 43],
            "parameters": {
                "collapse_level": [0.0, 0.5]
            }
        }

        base_path = tmp_path / "base.json"
        spec_path = tmp_path / "spec.json"

        with open(base_path, "w") as f:
            json.dump(base_config, f)

        with open(spec_path, "w") as f:
            json.dump(spec, f)

        out_runs = tmp_path / "runs"

        # Run generator
        subprocess.run([
            "python", "sweep/generate.py",
            "--base", str(base_path),
            "--spec", str(spec_path),
            "--out-dir", str(out_runs)
        ], check=True)

        # Verify generator output
        assert out_runs.exists()
        manifest_path = out_runs / "manifest.csv"
        assert manifest_path.exists()

        with open(manifest_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 5 # header + (2 collapse_levels * 2 seeds)

        # Verify specific dir
        dir_0_0_42 = out_runs / "test_sweep_collapse_level=0.0_seed42"
        assert dir_0_0_42.exists()
        assert (dir_0_0_42 / "config.json").exists()
        assert (dir_0_0_42 / "run.sh").exists()

        # 2. Mock results for collect
        # Seed 42, collapse 0.0: COMPLETE
        with open(dir_0_0_42 / "results.json", "w") as f:
            json.dump({
                "grokking_step": 50,
                "final_test_acc": 0.99,
                "final_weight_norm": 2.5
            }, f)

        # Seed 43, collapse 0.0: INCOMPLETE (bad json)
        dir_0_0_43 = out_runs / "test_sweep_collapse_level=0.0_seed43"
        with open(dir_0_0_43 / "results.json", "w") as f:
            f.write("bad json")

        # Seed 42, collapse 0.5: MISSING (no file)

        # Seed 43, collapse 0.5: COMPLETE (no grok)
        dir_0_5_43 = out_runs / "test_sweep_collapse_level=0.5_seed43"
        with open(dir_0_5_43 / "results.json", "w") as f:
            json.dump({
                "grokking_step": None,
                "final_test_acc": 0.50,
                "final_weight_norm": 5.0
            }, f)

        # 3. Collect
        csv_out = tmp_path / "results.csv"
        subprocess.run([
            "python", "sweep/collect.py",
            "--run-dir", str(out_runs),
            "--out-csv", str(csv_out)
        ], check=True)

        assert csv_out.exists()

        with open(csv_out, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4

        status_map = {f"{r['collapse_level']}_{r['seed']}": r['status'] for r in rows}
        assert status_map["0.0_42"] == "COMPLETE"
        assert status_map["0.0_43"] == "INCOMPLETE"
        assert status_map["0.5_42"] == "MISSING"
        assert status_map["0.5_43"] == "COMPLETE"

        # 4. Plot
        plot_out = tmp_path / "plots"
        subprocess.run([
            "python", "sweep/plot.py",
            "--results-csv", str(csv_out),
            "--x-axis", "collapse_level",
            "--out-dir", str(plot_out)
        ], check=True)

        # Verify plots were generated and are non-empty
        grok_plot = plot_out / "grok_vs_collapse_level.png"
        acc_plot = plot_out / "acc_vs_collapse_level.png"

        assert grok_plot.exists()
        assert grok_plot.stat().st_size > 0

        assert acc_plot.exists()
        assert acc_plot.stat().st_size > 0
