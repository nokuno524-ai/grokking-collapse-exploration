import math
from grokkit.cliff import find_cliff

def test_find_cliff_multiple_plateaus():
    runs = [
        {"wd": 0.1, "noise": 0.0, "final_fourier_concentration": 0.40},
        {"wd": 0.1, "noise": 0.1, "final_fourier_concentration": 0.38}, # plateau 1
        {"wd": 0.1, "noise": 0.2, "final_fourier_concentration": 0.25},
        {"wd": 0.1, "noise": 0.3, "final_fourier_concentration": 0.22}, # plateau 2
        {"wd": 0.1, "noise": 0.4, "final_fourier_concentration": 0.15}, # below threshold 0.20
    ]

    cliff = find_cliff(runs, "wd", "noise", "final_fourier_concentration", threshold=0.20, direction="below")

    assert math.isclose(cliff[0.1], 0.4)

def test_find_cliff_noisy_curve():
    runs = [
        {"wd": 0.1, "noise": 0.0, "final_fourier_concentration": 0.30},
        {"wd": 0.1, "noise": 0.1, "final_fourier_concentration": 0.18}, # dips below temporarily
        {"wd": 0.1, "noise": 0.2, "final_fourier_concentration": 0.25}, # jumps back up
        {"wd": 0.1, "noise": 0.3, "final_fourier_concentration": 0.10}, # drops below for good
    ]

    # Simple cliff detector returns first drop
    cliff = find_cliff(runs, "wd", "noise", "final_fourier_concentration", threshold=0.20, direction="below")
    assert math.isclose(cliff[0.1], 0.1)
