"""
Head-level circuit transplant and robustness analysis tools.
Varying eval seeds, layernorm recomputation, and checkpoint pairs.
"""

import argparse
from pathlib import Path

def check_robustness():
    """Robustness tools."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-seeds", nargs="+", type=int, help="Evaluation seeds")
    parser.add_argument("--pairs", nargs="+", type=Path, help="Pairs")
    args = parser.parse_args()
    print("Robustness tool.")

if __name__ == "__main__":
    check_robustness()
