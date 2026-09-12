"""
Generates the comprehensive multiplicity and robustness report.
"""
from pathlib import Path
import sys

from src.analysis.multiplicity_harness import run_harness, summarize_robustness

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found.")
        sys.exit(1)

    print("Running multiplicity harness...")
    regimes = run_harness(results_dir)

    report_md = summarize_robustness(regimes)

    out_path = Path("analysis/CLIFF_RIGOR.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(report_md)

    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    main()
