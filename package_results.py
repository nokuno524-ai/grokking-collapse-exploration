#!/usr/bin/env python3
"""
Results Packaging Script for Grokking-Collapse experiments.
Collects experiment outputs from results/ into a release-ready archive
(excluding raw checkpoints) and generates a markdown and PDF summary.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import tarfile
import markdown

try:
    import pdfkit
    HAS_PDFKIT = True
except ImportError:
    HAS_PDFKIT = False

def find_files(directory, exclude_ext=(".pt", ".pth")):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if not any(filename.endswith(ext) for ext in exclude_ext):
                files.append(Path(root) / filename)
    return files

def generate_summary(results_dir, output_md, output_pdf):
    results_dir = Path(results_dir)
    markdown_content = ["# Grokking-Collapse Experiment Results Summary\n\n"]

    # Try to summarize phase1 basic grokking if available
    phase1_dir = results_dir / "phase1"
    if phase1_dir.exists():
        markdown_content.append("## Phase 1: Basic Grokking Outcomes\n\n")
        markdown_content.append("| Condition | Grokked | Grokking Step | Final Test Acc | Final Fourier |\n")
        markdown_content.append("|---|---|---|---|---|\n")

        for condition in sorted(os.listdir(phase1_dir)):
            cond_dir = phase1_dir / condition
            res_file = cond_dir / "results.json"
            if res_file.exists():
                with open(res_file) as f:
                    data = json.load(f)
                    markdown_content.append(
                        f"| {condition} | {data.get('grokked')} | {data.get('grokking_step')} "
                        f"| {data.get('final_test_acc', 0):.4f} | {data.get('final_fourier_concentration', 0):.4f} |\n"
                    )
        markdown_content.append("\n")

    md_text = "".join(markdown_content)
    with open(output_md, 'w') as f:
        f.write(md_text)

    if HAS_PDFKIT:
        try:
            html = markdown.markdown(md_text, extensions=['tables'])
            # Add some basic CSS for tables
            html = f"""
            <html>
            <head>
            <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            </style>
            </head>
            <body>
            {html}
            </body>
            </html>
            """
            pdfkit.from_string(html, output_pdf, options={'enable-local-file-access': None})
            print(f"Generated PDF summary: {output_pdf}")
        except Exception as e:
            print(f"Warning: Could not generate PDF. Ensure wkhtmltopdf is installed. Error: {e}")
    else:
        print("Warning: pdfkit not installed. Skipping PDF generation.")

def create_readme(output_file):
    readme_content = """# Results Package

This package contains the processed results, figures, and summary data from the Grokking-Collapse reproduction runs.
Large checkpoint files (`.pt`) are excluded to keep the package size manageable.

## Contents
- `results_summary.md`: A top-level summary of the outcomes for Phase 1.
- `results_summary.pdf`: PDF version of the summary.
- `results/`: The collected JSON and plot outputs from each phase of the pipeline.
"""
    with open(output_file, 'w') as f:
        f.write(readme_content)

def package_results(results_dir, output_tar, summary_file_md, summary_file_pdf, readme_file):
    print(f"Collecting files from {results_dir} (excluding .pt checkpoints)...")

    # Generate summaries
    generate_summary(results_dir, summary_file_md, summary_file_pdf)
    create_readme(readme_file)

    # Collect files
    files_to_pack = find_files(results_dir)

    print(f"Creating archive {output_tar}...")
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(summary_file_md, arcname=os.path.basename(summary_file_md))
        if os.path.exists(summary_file_pdf):
            tar.add(summary_file_pdf, arcname=os.path.basename(summary_file_pdf))
        tar.add(readme_file, arcname=os.path.basename(readme_file))
        for f in files_to_pack:
            tar.add(f, arcname=str(f))

    print("Packaging complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_reproduce", help="Directory with results")
    parser.add_argument("--output-tar", type=str, default="grokking_collapse_results.tar.gz")
    parser.add_argument("--summary-file-md", type=str, default="results_summary.md")
    parser.add_argument("--summary-file-pdf", type=str, default="results_summary.pdf")
    parser.add_argument("--readme-file", type=str, default="RESULTS_README.md")

    args = parser.parse_args()

    # Check if results exist
    if not os.path.exists(args.results_dir):
        print(f"Warning: Directory {args.results_dir} does not exist. The package will only contain summaries of empty directories.")
        os.makedirs(args.results_dir, exist_ok=True)

    package_results(args.results_dir, args.output_tar, args.summary_file_md, args.summary_file_pdf, args.readme_file)
