import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

def parse_txt_log(file_path: Path) -> Dict[str, Any]:
    """Parse metrics from a plain text log file."""
    # Step   100 | train_loss=3.5744 test_loss=4.5381 | train_acc=0.1552 test_acc=0.0119 | ‖W‖=25.23 rank=52.2 fourier=0.098 | time=1.1s
    history = []

    with open(file_path, "r") as f:
        content = f.read()

    # Try to extract condition and collapse level
    condition_name = "unknown"
    collapse_level = 0.0

    cond_match = re.search(r"Condition:\s*([^,]+),\s*collapse_level=([\d\.]+)", content)
    if cond_match:
        condition_name = cond_match.group(1).strip()
        collapse_level = float(cond_match.group(2))

    # Extract steps
    step_pattern = re.compile(
        r"Step\s+(?P<step>\d+)\s*\|\s*train_loss=(?P<train_loss>[\d\.]+)\s+test_loss=(?P<test_loss>[\d\.]+)\s*\|\s*"
        r"train_acc=(?P<train_acc>[\d\.]+)\s+test_acc=(?P<test_acc>[\d\.]+)\s*\|\s*"
        r"‖W‖=(?P<weight_norm>[\d\.]+)\s+rank=(?P<rank>[\d\.]+)\s+fourier=(?P<fourier>[\d\.]+)"
    )

    for match in step_pattern.finditer(content):
        step_data = match.groupdict()
        # Convert to appropriate types
        history.append({
            "step": int(step_data["step"]),
            "train_loss": float(step_data["train_loss"]),
            "test_loss": float(step_data["test_loss"]),
            "train_acc": float(step_data["train_acc"]),
            "test_acc": float(step_data["test_acc"]),
            "weight_norm": float(step_data["weight_norm"]),
            "embedding_rank": float(step_data["rank"]),
            "fourier_concentration": float(step_data["fourier"]),
        })

    result = {
        "config": {
            "condition_name": condition_name,
            "collapse_level": collapse_level,
        },
        "history": history
    }

    if history:
        last = history[-1]
        result.update({
            "final_train_acc": last["train_acc"],
            "final_test_acc": last["test_acc"],
            "final_weight_norm": last["weight_norm"],
            "final_embedding_rank": last["embedding_rank"],
            "final_fourier_concentration": last["fourier_concentration"],
        })

        # Estimate grokking step (first step > 0.95 test acc)
        grokking_step = -1
        for h in history:
            if h["test_acc"] > 0.95:
                grokking_step = h["step"]
                break

        result["grokking_step"] = grokking_step if grokking_step != -1 else 0
        result["grokked"] = grokking_step != -1

    return result

def parse_wandb_log(file_path: Path) -> Dict[str, Any]:
    """Parse metrics from a wandb run export json."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Standard wandb export format adaptation
        history = []
        if isinstance(data, list):
            history = data
        elif "history" in data:
            history = data["history"]
        elif "systemMetrics" in data or "summaryMetrics" in data:
            # Full run export
            history = data.get("history", [])

        # Clean wandb specific keys
        clean_history = []
        for row in history:
            clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
            if clean_row:
                clean_history.append(clean_row)

        return {"history": clean_history}
    except Exception as e:
        print(f"Error parsing wandb log {file_path}: {e}")
        return {}

def parse_tensorboard_log(file_path: Path) -> Dict[str, Any]:
    """Parse metrics from tensorboard events file using tensorboard (if available)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not installed. Cannot parse TB logs.")
        return {}

    try:
        event_acc = EventAccumulator(str(file_path))
        event_acc.Reload()

        history = []
        tags = event_acc.Tags()['scalars']

        # Determine steps
        if not tags:
            return {}

        steps = [s.step for s in event_acc.Scalars(tags[0])]

        for i, step in enumerate(steps):
            step_data = {"step": step}
            for tag in tags:
                try:
                    step_data[tag] = event_acc.Scalars(tag)[i].value
                except IndexError:
                    pass
            history.append(step_data)

        return {"history": history}
    except Exception as e:
        print(f"Error parsing tensorboard log {file_path}: {e}")
        return {}

def parse_results_dir(results_dir: Path) -> List[Dict[str, Any]]:
    """Recursively parse all result files in a directory."""
    parsed_runs = []

    # 1. Parse results.json files
    for json_file in results_dir.rglob("results.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Flatten config into top level for easier tabular representation
            run_data = {
                "source_file": str(json_file),
                "run_id": f"{json_file.parent.name}_{json_file.parent.parent.name}"
            }

            # Add top level metrics
            for k, v in data.items():
                if k != "config" and k != "history" and not isinstance(v, (dict, list)):
                    run_data[k] = v

            # Add config
            if "config" in data:
                for k, v in data["config"].items():
                    run_data[f"config_{k}"] = v

            # Add final history step if available and metrics aren't present
            if "history" in data and len(data["history"]) > 0:
                last_step = data["history"][-1]
                run_data["max_steps_run"] = last_step.get("step", 0)

                # Compute area under curve for test_acc as an extra metric
                test_accs = [h.get("test_acc", 0) for h in data["history"]]
                run_data["auc_test_acc"] = sum(test_accs) / len(test_accs) if test_accs else 0

            parsed_runs.append(run_data)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    # 2. Parse text logs
    if results_dir.name == "logs" or results_dir.parent.name == "logs":
        for txt_file in results_dir.rglob("*.out"):
            try:
                data = parse_txt_log(txt_file)
                if data and "history" in data and len(data["history"]) > 0:
                    run_data = {
                        "source_file": str(txt_file),
                        "run_id": txt_file.stem
                    }

                    for k, v in data.items():
                        if k != "config" and k != "history" and not isinstance(v, (dict, list)):
                            run_data[k] = v

                    if "config" in data:
                        for k, v in data["config"].items():
                            run_data[f"config_{k}"] = v

                    parsed_runs.append(run_data)
            except Exception as e:
                print(f"Error parsing text log {txt_file}: {e}")


    # 3. Parse TensorBoard logs
    for tb_file in results_dir.rglob("events.out.tfevents.*"):
        try:
            tb_data = parse_tensorboard_log(tb_file)
            if tb_data and "history" in tb_data and tb_data["history"]:
                run_data = {
                    "source_file": str(tb_file),
                    "run_id": tb_file.parent.name
                }

                last_step = tb_data["history"][-1]
                run_data["max_steps_run"] = last_step.get("step", 0)

                for k, v in last_step.items():
                    if k != "step":
                        run_data[f"final_{k}"] = v

                parsed_runs.append(run_data)
        except Exception as e:
            print(f"Error parsing tensorboard log {tb_file}: {e}")

    # 4. Parse wandb logs
    for wandb_file in results_dir.rglob("wandb_export*.json"):
        try:
            w_data = parse_wandb_log(wandb_file)
            if w_data and "history" in w_data and w_data["history"]:
                run_data = {
                    "source_file": str(wandb_file),
                    "run_id": wandb_file.stem
                }

                last_step = w_data["history"][-1]
                run_data["max_steps_run"] = last_step.get("step", 0)

                for k, v in last_step.items():
                    if k != "step":
                        run_data[f"final_{k}"] = v

                parsed_runs.append(run_data)
        except Exception as e:
            print(f"Error parsing wandb log {wandb_file}: {e}")

    return parsed_runs

def export_to_csv(data: List[Dict[str, Any]], output_path: Path):
    """Export parsed runs to a CSV file."""
    if not data:
        print("No data to export to CSV.")
        return

    # Get all unique keys across all runs
    fieldnames = set()
    for run in data:
        fieldnames.update(run.keys())

    fieldnames = sorted(list(fieldnames))

    # Make sure source_file and run_id are first
    for field in ["run_id", "source_file", "config_condition_name", "config_seed", "grokked", "grokking_step"]:
        if field in fieldnames:
            fieldnames.remove(field)
            fieldnames.insert(0, field)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in data:
            writer.writerow(run)

def export_to_json(data: List[Dict[str, Any]], output_path: Path):
    """Export parsed runs to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Parse experiment results into structured formats.")
    parser.add_argument("results_dir", type=str, nargs="?", default="results", help="Directory containing results")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory containing text logs")
    parser.add_argument("--output-csv", type=str, default="parsed_results.csv", help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="parsed_results.json", help="Output JSON path")

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    logs_path = Path(args.logs_dir)

    print(f"Parsing results from {results_path}...")
    parsed_data = parse_results_dir(results_path)

    if logs_path.exists() and logs_path.is_dir():
        print(f"Parsing logs from {logs_path}...")
        parsed_data.extend(parse_results_dir(logs_path))

    print(f"Found {len(parsed_data)} valid runs.")

    if parsed_data:
        export_to_csv(parsed_data, Path(args.output_csv))
        export_to_json(parsed_data, Path(args.output_json))
        print(f"Exported to {args.output_csv} and {args.output_json}")
    else:
        print("No results found. Export skipped.")

if __name__ == "__main__":
    main()
