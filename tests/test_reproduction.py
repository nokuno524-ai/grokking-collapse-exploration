import pytest
import subprocess
import os

def test_run_phase_script_exists():
    assert os.path.exists('reproduce/run_phase.sh')

def test_run_all_script_exists():
    assert os.path.exists('reproduce/run_all.sh')

def test_run_phase_invalid_phase():
    """Test that running with an invalid phase returns error code."""
    # Run in a subprocess
    result = subprocess.run(['bash', 'reproduce/run_phase.sh', '99'], capture_output=True)
    assert result.returncode != 0
    assert b"Unknown phase: 99" in result.stdout

def test_run_phase_no_args():
    result = subprocess.run(['bash', 'reproduce/run_phase.sh'], capture_output=True)
    assert result.returncode != 0
    assert b"Usage: ./run_phase.sh <phase_number>" in result.stdout
