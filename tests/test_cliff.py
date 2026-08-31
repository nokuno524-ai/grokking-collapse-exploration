import math
from grokkit.cliff import find_cliff

def test_find_cliff():
    runs = [
        {"wd": 0.1, "noise": 0.0, "final_fourier_concentration": 0.25},
        {"wd": 0.1, "noise": 0.1, "final_fourier_concentration": 0.15}, # below threshold 0.20
        {"wd": 0.2, "noise": 0.0, "final_fourier_concentration": 0.30},
        {"wd": 0.2, "noise": 0.1, "final_fourier_concentration": 0.25},
        {"wd": 0.2, "noise": 0.2, "final_fourier_concentration": 0.10}, # below threshold 0.20
    ]

    cliff = find_cliff(runs, "wd", "noise", "final_fourier_concentration", threshold=0.20, direction="below")

    assert math.isclose(cliff[0.1], 0.1)
    assert math.isclose(cliff[0.2], 0.2)

def test_find_cliff_no_drop():
    runs = [
        {"wd": 0.1, "noise": 0.0, "final_fourier_concentration": 0.25},
        {"wd": 0.1, "noise": 0.1, "final_fourier_concentration": 0.22},
    ]

    cliff = find_cliff(runs, "wd", "noise", "final_fourier_concentration", threshold=0.20, direction="below")

    assert cliff[0.1] is None
