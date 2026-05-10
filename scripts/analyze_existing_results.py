#!/usr/bin/env python3
import argparse
import pathlib
import sys

# Ensure src is in the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.analysis.parser import scan_results_dir
from src.analysis.visualizer import generate_experiment_report

def main():
    parser = argparse.ArgumentParser(description="Analyze existing experiment results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="results/analysis_report", help="Output directory for the analysis report")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)

    print(f"Scanning results directory: {results_dir}")
    catalog = scan_results_dir(str(results_dir))

    if not catalog:
        print("No valid experiment results found.")
        sys.exit(0)

    print(f"Found {len(catalog)} valid experiment conditions:")
    for entry in catalog:
        print(f"  - {entry['condition_name']} (Grokked: {entry['grokked']}, Acc: {entry['final_test_acc']:.4f})")

    output_dir = pathlib.Path(args.output_dir)
    print(f"\nGenerating experiment report in: {output_dir}")
    generate_experiment_report(str(results_dir), str(output_dir))

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
