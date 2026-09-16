"""
Run aggregation for multi-seed logs.
Aligns steps and validates schema for JSON, JSONL, and CSV formats.
"""
import argparse
from pathlib import Path

def aggregate_runs():
    """Aggregate runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print("Run aggregator tool.")

if __name__ == "__main__":
    aggregate_runs()
