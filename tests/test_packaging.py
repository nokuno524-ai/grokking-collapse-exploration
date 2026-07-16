import pytest
import os
import tarfile
from package_results import find_files, package_results

def test_find_files_excludes_checkpoints(tmp_path):
    # Create mock directory structure
    d = tmp_path / "results"
    d.mkdir()
    (d / "model.pt").write_text("checkpoint")
    (d / "results.json").write_text("{}")

    files = find_files(d)
    assert len(files) == 1
    assert "results.json" in str(files[0])
    assert "model.pt" not in [str(f) for f in files]

def test_package_results_creates_archive(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.json").write_text('{"grokked": true}')

    tar_out = tmp_path / "out.tar.gz"
    summary_md = tmp_path / "summary.md"
    summary_pdf = tmp_path / "summary.pdf"
    readme = tmp_path / "README.md"

    package_results(str(results_dir), str(tar_out), str(summary_md), str(summary_pdf), str(readme))

    assert tar_out.exists()
    assert summary_md.exists()
    assert readme.exists()

    # Check archive content
    with tarfile.open(tar_out, "r:gz") as tar:
        names = tar.getnames()
        assert any("summary.md" in n for n in names)
        assert any("README.md" in n for n in names)
        assert any("results.json" in n for n in names)
