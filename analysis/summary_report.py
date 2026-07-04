import json
import argparse
from pathlib import Path

def generate_summary_report(results_dir="results", output_file="analysis/summary_report.md"):
    run_paths = [p for p in Path(results_dir).iterdir() if p.is_dir()]

    report_lines = [
        "# Experiment Summary Report",
        "",
        "This report aggregates the results from the collapse vs grokking experiments.",
        "",
        "## Overall Results",
        "",
        "| Condition | Grokking Step | Final Train Acc | Final Test Acc | Final Weight Norm | Final Fourier Concentration |",
        "|---|---|---|---|---|---|"
    ]

    for run_path in sorted(run_paths):
        res_file = run_path / "results.json"
        if not res_file.exists():
            continue

        with open(res_file, "r") as f:
            data = json.load(f)

        grokked = data.get("grokked", False)
        step = str(data.get("grokking_step", "N/A")) if grokked else "Did not grok"

        train_acc = data.get("final_train_acc", 0.0)
        test_acc = data.get("final_test_acc", 0.0)
        weight_norm = data.get("final_weight_norm", 0.0)
        fourier = data.get("final_fourier_concentration", 0.0)

        report_lines.append(
            f"| {run_path.name} | {step} | {train_acc:.4f} | {test_acc:.4f} | {weight_norm:.2f} | {fourier:.4f} |"
        )

    report_lines.append("")
    report_lines.append("## Detailed Configuration Reference")
    report_lines.append("")

    for run_path in sorted(run_paths):
        res_file = run_path / "results.json"
        if not res_file.exists():
            continue

        with open(res_file, "r") as f:
            data = json.load(f)
            config = data.get("config", {})

        report_lines.append(f"### {run_path.name}")
        report_lines.append("```json")
        report_lines.append(json.dumps(config, indent=2))
        report_lines.append("```")
        report_lines.append("")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Summary report generated at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-file", type=str, default="analysis/summary_report.md")
    args = parser.parse_args()
    generate_summary_report(args.results_dir, args.output_file)
