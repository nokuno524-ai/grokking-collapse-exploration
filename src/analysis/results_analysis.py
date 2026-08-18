import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def detect_grokking_step(accuracies: List[float], threshold: float = 0.95, window_size: int = 5) -> Optional[int]:
    """
    Detect the step where the model "groks", defined as maintaining accuracy above
    the threshold for a continuous window of steps.

    Args:
        accuracies: List of accuracy values across training steps.
        threshold: The accuracy threshold defining grokking.
        window_size: Number of consecutive steps required to stay above threshold.

    Returns:
        The step index (0-based relative to the list) where grokking was first achieved
        and maintained for `window_size` steps. Returns None if it never groks.
    """
    if len(accuracies) < window_size:
        return None

    for i in range(len(accuracies) - window_size + 1):
        window = accuracies[i:i+window_size]
        if all(acc >= threshold for acc in window):
            return i

    return None


def parse_results_dir(results_dir: Path, window_size: int = 5, threshold: float = 0.95) -> pd.DataFrame:
    """
    Parse all results.json files in a directory recursively.

    Args:
        results_dir: Path to the results directory.
        window_size: Window size for grokking detection.
        threshold: Accuracy threshold for grokking.

    Returns:
        DataFrame containing parsed metrics for each run.
    """
    data = []

    for json_file in results_dir.rglob("results.json"):
        try:
            with open(json_file, 'r') as f:
                res = json.load(f)

            config = res.get("config", {})
            history = res.get("history", [])

            if not history:
                continue

            steps = [h.get("step", i) for i, h in enumerate(history)]
            test_accs = [h.get("test_acc", 0.0) for h in history]

            grok_idx = detect_grokking_step(test_accs, threshold=threshold, window_size=window_size)
            grok_step = steps[grok_idx] if grok_idx is not None else -1

            entry = {
                "run_dir": str(json_file.parent),
                "condition": config.get("condition_name", "unknown"),
                "collapse_level": config.get("collapse_level", 0.0),
                "collapse_severity": config.get("collapse_severity", 0.0),
                "final_test_acc": res.get("final_test_acc", test_accs[-1] if test_accs else 0.0),
                "final_train_acc": res.get("final_train_acc", history[-1].get("train_acc", 0.0) if history else 0.0),
                "final_weight_norm": res.get("final_weight_norm", history[-1].get("weight_norm", 0.0) if history else 0.0),
                "grokking_step_detected": grok_step,
                "grokked_detected": grok_step != -1,
                "seed": config.get("seed", 42),
            }
            data.append(entry)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    return pd.DataFrame(data)


def plot_loss_curves(df: pd.DataFrame, results_dir: Path, output_path: Path):
    """
    Plots training and testing loss curves for aggregated results.

    Args:
        df: The dataframe of parsed runs.
        results_dir: Base directory of results (unused, kept for API compatibility).
        output_path: Where to save the output plot PNG.
    """
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad"
    }

    for _, row in df.iterrows():
        json_file = Path(row["run_dir"]) / "results.json"
        try:
            with open(json_file, 'r') as f:
                res = json.load(f)

            history = res.get("history", [])
            if not history:
                continue

            steps = [h.get("step") for h in history]
            train_loss = [h.get("train_loss") for h in history]
            test_loss = [h.get("test_loss") for h in history]

            cond = row["condition"]
            color = colors.get(cond, "gray")

            label = cond if cond not in [l.get_label() for l in ax_train.lines] else None

            ax_train.plot(steps, train_loss, color=color, alpha=0.7, label=label)
            ax_test.plot(steps, test_loss, color=color, alpha=0.7, label=label)
        except Exception as e:
            print(f"Error plotting {json_file}: {e}")

    ax_train.set_title("Training Loss vs Steps")
    ax_train.set_xlabel("Steps")
    ax_train.set_ylabel("Cross Entropy Loss")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend()

    ax_test.set_title("Testing Loss vs Steps")
    ax_test.set_xlabel("Steps")
    ax_test.set_ylabel("Cross Entropy Loss")
    ax_test.grid(True, alpha=0.3)
    ax_test.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    """Main function for processing result logs."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-csv", type=str, default="analysis/aggregated_results.csv")
    parser.add_argument("--output-plot", type=str, default="analysis/loss_curves_summary.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_csv = Path(args.output_csv)
    out_plot = Path(args.output_plot)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = parse_results_dir(results_dir)
    if not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"Aggregated {len(df)} runs to {out_csv}")
        plot_loss_curves(df, results_dir, out_plot)
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
