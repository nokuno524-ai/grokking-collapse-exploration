"""Parse and summarize experiment results.

This script parses training logs (JSON or text), computes summary statistics,
and generates a text-based comparison table across conditions.
"""

import json
import os
import re

def parse_training_log(filepath: str) -> dict:
    """Parse a training log file (JSON or text) and extract step, loss, and accuracy.

    Args:
        filepath: Path to the log file (.json or .out/.log).

    Returns:
        A dictionary containing the parsed data, typically a list of steps with metrics.
        The format returned is: {'history': [{'step': int, 'train_loss': float, 'test_loss': float, 'train_acc': float, 'test_acc': float}, ...]}
    """
    data = {'history': []}

    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                content = json.load(f)
                if 'history' in content:
                    # Filter history to only required fields to be consistent
                    for entry in content['history']:
                        data['history'].append({
                            'step': entry.get('step', 0),
                            'train_loss': entry.get('train_loss', 0.0),
                            'test_loss': entry.get('test_loss', 0.0),
                            'train_acc': entry.get('train_acc', 0.0),
                            'test_acc': entry.get('test_acc', 0.0)
                        })
        else:
            with open(filepath, 'r') as f:
                # Regex to match log lines like:
                # Step   100 | train_loss=3.5744 test_loss=4.5381 | train_acc=0.1552 test_acc=0.0119 | ...
                pattern = re.compile(
                    r"Step\s+(?P<step>\d+)\s*\|\s*train_loss=(?P<train_loss>[\d.]+)\s+test_loss=(?P<test_loss>[\d.]+)\s*\|\s*train_acc=(?P<train_acc>[\d.]+)\s+test_acc=(?P<test_acc>[\d.]+)"
                )
                for line in f:
                    match = pattern.search(line)
                    if match:
                        data['history'].append({
                            'step': int(match.group('step')),
                            'train_loss': float(match.group('train_loss')),
                            'test_loss': float(match.group('test_loss')),
                            'train_acc': float(match.group('train_acc')),
                            'test_acc': float(match.group('test_acc'))
                        })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return data

def compute_summary(data: dict) -> dict:
    """Compute summary metrics from parsed training data.

    Calculates min loss (test_loss), max accuracy (test_acc), step of best accuracy,
    and convergence step (step where test_acc >= 0.95).

    Args:
        data: Dictionary containing 'history' list of metrics per step.

    Returns:
        Dictionary with summary metrics.
    """
    history = data.get('history', [])
    if not history:
        return {
            'min_loss': None,
            'max_accuracy': None,
            'step_of_best_accuracy': None,
            'convergence_step': None
        }

    min_loss = float('inf')
    max_accuracy = -float('inf')
    step_of_best_accuracy = None
    convergence_step = None

    for entry in history:
        step = entry.get('step', 0)
        test_loss = entry.get('test_loss', float('inf'))
        test_acc = entry.get('test_acc', 0.0)

        if test_loss < min_loss:
            min_loss = test_loss

        if test_acc > max_accuracy:
            max_accuracy = test_acc
            step_of_best_accuracy = step

        if convergence_step is None and test_acc >= 0.95:
            convergence_step = step

    return {
        'min_loss': min_loss if min_loss != float('inf') else None,
        'max_accuracy': max_accuracy if max_accuracy != -float('inf') else None,
        'step_of_best_accuracy': step_of_best_accuracy,
        'convergence_step': convergence_step
    }

def compare_conditions(results: dict) -> str:
    """Generate a text-based comparison table from multiple conditions' summaries.

    Args:
        results: Dictionary mapping condition names to their summary dictionaries.

    Returns:
        Formatted string containing the comparison table.
    """
    if not results:
        return "No results to compare."

    # Define table headers
    headers = ["Condition", "Min Loss", "Max Acc", "Best Step", "Conv Step"]

    # Calculate column widths based on headers and data
    col_widths = [len(h) for h in headers]

    rows = []
    for condition, summary in results.items():
        min_loss = f"{summary.get('min_loss'):.4f}" if summary.get('min_loss') is not None else "N/A"
        max_acc = f"{summary.get('max_accuracy'):.4f}" if summary.get('max_accuracy') is not None else "N/A"
        best_step = str(summary.get('step_of_best_accuracy')) if summary.get('step_of_best_accuracy') is not None else "N/A"
        conv_step = str(summary.get('convergence_step')) if summary.get('convergence_step') is not None else "N/A"

        row = [condition, min_loss, max_acc, best_step, conv_step]
        rows.append(row)

        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    # Format table
    separator = "-" * (sum(col_widths) + len(col_widths) * 3 + 1)

    table = []
    table.append(separator)
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    table.append(header_row)
    table.append(separator)

    for row in rows:
        formatted_row = "| " + " | ".join(val.ljust(w) for val, w in zip(row, col_widths)) + " |"
        table.append(formatted_row)

    table.append(separator)

    return "\n".join(table)

def main():
    """Main function to walk results directory, parse logs, and print comparison."""
    results_dir = 'results'
    all_summaries = {}

    if not os.path.exists(results_dir):
        print(f"Directory '{results_dir}' not found.")
        return

    # Look for results.json in subdirectories of results_dir
    for condition in os.listdir(results_dir):
        cond_dir = os.path.join(results_dir, condition)
        if os.path.isdir(cond_dir):
            json_path = os.path.join(cond_dir, 'results.json')
            if os.path.exists(json_path):
                data = parse_training_log(json_path)
                summary = compute_summary(data)
                all_summaries[condition] = summary

    # Also check logs directory for text logs just in case there are other conditions
    logs_dir = 'logs'
    if os.path.exists(logs_dir):
        for log_file in os.listdir(logs_dir):
            if log_file.endswith('.out') or log_file.endswith('.log'):
                log_path = os.path.join(logs_dir, log_file)
                data = parse_training_log(log_path)
                if data['history']:
                    # Use filename as condition name for text logs if not already in summaries
                    condition_name = os.path.splitext(log_file)[0]
                    if condition_name not in all_summaries:
                        summary = compute_summary(data)
                        all_summaries[condition_name] = summary

    if all_summaries:
        print("Experiment Results Comparison:")
        print(compare_conditions(all_summaries))
    else:
        print("No valid log files found to analyze.")

if __name__ == "__main__":
    main()
