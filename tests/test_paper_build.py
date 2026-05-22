import pytest
import subprocess
import shutil
from pathlib import Path

def test_latex_compilation():
    if not shutil.which("pdflatex"):
        pytest.skip("pdflatex is not installed")

    paper_dir = Path(__file__).parent.parent / "paper"

    # Run simple make
    result = subprocess.run(["make"], cwd=paper_dir, capture_output=True, text=True)
    assert result.returncode == 0, f"LaTeX build failed:\n{result.stderr}"

    assert (paper_dir / "main.pdf").exists()

    # Run clean
    subprocess.run(["make", "clean"], cwd=paper_dir)
    assert not (paper_dir / "main.pdf").exists()
