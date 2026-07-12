import os
import json
import csv
import pandas as pd
from tabulate import tabulate
from pathlib import Path

def analyze_results(results_dir="results"):
    data = []

    # Iterate over main subdirectories in results_dir (pure, low_collapse, etc.)
    base_path = Path(results_dir)

    for condition_dir in base_path.iterdir():
        if condition_dir.is_dir():
            json_path = condition_dir / "results.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    try:
                        res = json.load(f)
                    except json.JSONDecodeError:
                        continue

                condition_name = res.get("config", {}).get("condition_name", condition_dir.name)
                grokked = res.get("grokked", False)
                grokking_step = res.get("grokking_step", None)
                final_test_acc = res.get("final_test_acc", 0.0)
                final_train_acc = res.get("final_train_acc", 0.0)
                final_weight_norm = res.get("final_weight_norm", 0.0)
                final_embedding_rank = res.get("final_embedding_rank", 0.0)
                final_fourier_concentration = res.get("final_fourier_concentration", 0.0)

                history = res.get("history", [])
                max_test_acc = max([h.get("test_acc", 0) for h in history]) if history else final_test_acc
                max_weight_norm = max([h.get("weight_norm", 0) for h in history]) if history else final_weight_norm
                max_embedding_rank = max([h.get("embedding_rank", 0) for h in history]) if history else final_embedding_rank

                data.append({
                    "Condition": condition_name,
                    "Grokked": grokked,
                    "Grokking Step": grokking_step if grokking_step is not None else "N/A",
                    "Final Train Acc": f"{final_train_acc:.4f}",
                    "Final Test Acc": f"{final_test_acc:.4f}",
                    "Max Test Acc": f"{max_test_acc:.4f}",
                    "Final Weight Norm": f"{final_weight_norm:.4f}",
                    "Max Weight Norm": f"{max_weight_norm:.4f}",
                    "Final Embed Rank": f"{final_embedding_rank:.4f}",
                    "Max Embed Rank": f"{max_embedding_rank:.4f}",
                    "Fourier Conc": f"{final_fourier_concentration:.4f}"
                })

    if not data:
        print("No results.json found in subdirectories of", results_dir)
        return

    df = pd.DataFrame(data)

    # Sort logically by predefined order if possible
    order = {"pure": 0, "low_collapse": 1, "medium_collapse": 2, "high_collapse": 3, "severe_collapse": 4}
    df['sort_key'] = df['Condition'].map(lambda x: order.get(x, 99))
    df = df.sort_values('sort_key').drop('sort_key', axis=1)

    print("\nExperiment Results Summary:\n")
    print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

    # Save to CSV
    output_csv = base_path / "comprehensive_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved comprehensive summary to {output_csv}")

    # Save to MD
    output_md = base_path / "ANALYSIS.md"
    with open(output_md, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n")
    print(f"Saved comprehensive summary markdown to {output_md}")


if __name__ == "__main__":
    analyze_results()
