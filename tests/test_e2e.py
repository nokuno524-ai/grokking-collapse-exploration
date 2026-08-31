import subprocess
from pathlib import Path

def test_e2e_markdown_report():
    # Use the fixture created earlier
    fixture_dir = Path("tests/fixtures/runs")

    res = subprocess.run(["grokkit", "analyze", str(fixture_dir)], capture_output=True, text=True)
    assert res.returncode == 0

    out = res.stdout
    assert "| cond1 |" in out
    assert "| 0.990 |" in out
    assert "| 1000 |" in out
