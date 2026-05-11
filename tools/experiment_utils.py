import dataclasses
import json
import urllib.parse
import os
import math
from typing import Dict, Any, List


@dataclasses.dataclass
class ExperimentConfig:
    name: str
    collapse_level: float
    model_size: int
    dataset: str
    learning_rate: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**d)

    def to_filename(self) -> str:
        """Generates a safe filename based on the configuration."""
        parts = [
            f"name={self.name}",
            f"collapse={self.collapse_level}",
            f"size={self.model_size}",
            f"data={self.dataset}",
            f"lr={self.learning_rate}",
            f"seed={self.seed}",
        ]
        filename = "_".join(parts)
        # Urlencode to ensure safe characters
        return urllib.parse.quote(filename, safe="=-_")


def generate_sweep(base_config: dict, param_name: str, values: list) -> List[ExperimentConfig]:
    """Generates a list of ExperimentConfigs by sweeping over a parameter."""
    configs = []
    for val in values:
        current_config = base_config.copy()
        current_config[param_name] = val
        configs.append(ExperimentConfig.from_dict(current_config))
    return configs


def aggregate_results(results_dir: str) -> dict:
    """Scans for JSON result files, groups by config, and computes stats."""
    import collections

    # Structure to hold results: Grouped by a hashable representation of config (without seed)
    # Using tuple of sorted items (excluding seed) as key
    raw_data = collections.defaultdict(lambda: {
        "final_accuracy": [],
        "grokking_step": [],
        "weight_norm_change": []
    })

    # Explore directory for results.json files
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file == "results.json" or file.endswith(".json"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                        # We need config, final_test_acc, grokking_step, final_weight_norm, history[0].weight_norm
                        # Using exact fields requested in task or closest match from exploration

                        config_data = data.get("config", {})

                        # We only want to group by ExperimentConfig equivalent fields, but ignoring seed.
                        # Since files might have generic configs (like the pure one), we map to ExperimentConfig fields if possible.
                        # If a config doesn't exactly match ExperimentConfig, we extract what we can or fall back.
                        exp_config_fields = {}
                        if isinstance(config_data, dict):
                            # Try to extract standard ExperimentConfig fields or keep original config data
                            exp_config_fields = {k: v for k, v in config_data.items() if k != "seed"}

                        group_key = tuple(sorted(exp_config_fields.items(), key=lambda x: x[0]))

                        # Extract metrics
                        final_acc = data.get("final_test_acc")
                        if final_acc is None:
                            final_acc = data.get("final_accuracy") # Fallback

                        grokking_step = data.get("grokking_step")

                        final_wn = data.get("final_weight_norm")
                        initial_wn = None
                        if "history" in data and len(data["history"]) > 0:
                            initial_wn = data["history"][0].get("weight_norm")

                        wn_change = None
                        if final_wn is not None and initial_wn is not None:
                            wn_change = final_wn - initial_wn

                        # Only add if we have some data
                        if final_acc is not None:
                            raw_data[group_key]["final_accuracy"].append(final_acc)
                        if grokking_step is not None:
                            raw_data[group_key]["grokking_step"].append(grokking_step)
                        if wn_change is not None:
                            raw_data[group_key]["weight_norm_change"].append(wn_change)

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    # Compute mean and std
    aggregated = {}

    def compute_stats(values):
        if not values:
            return {"mean": None, "std": None}
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return {"mean": mean, "std": std}

    for group_key, metrics in raw_data.items():
        # Convert tuple of tuples back to dict
        config_dict = dict(group_key)

        # Stringify config_dict safely to use as a key in JSON-like output, or keep it as string representation
        # It's better to store string representation of config dict as key
        str_key = str(config_dict)

        aggregated[str_key] = {
            "config_without_seed": config_dict,
            "metrics": {
                "final_accuracy": compute_stats(metrics["final_accuracy"]),
                "grokking_step": compute_stats(metrics["grokking_step"]),
                "weight_norm_change": compute_stats(metrics["weight_norm_change"]),
                "num_seeds": max(
                    len(metrics["final_accuracy"]),
                    len(metrics["grokking_step"]),
                    len(metrics["weight_norm_change"])
                )
            }
        }

    return aggregated


def compare_conditions(results: dict) -> str:
    """Creates a Markdown table comparing all conditions from aggregated results."""
    if not results:
        return "No results to compare."

    lines = []
    lines.append("| Condition | Num Seeds | Final Acc (Mean ± Std) | Grokking Step (Mean ± Std) | Weight Norm Change (Mean ± Std) |")
    lines.append("|---|---|---|---|---|")

    for str_key, item in results.items():
        config = item["config_without_seed"]
        metrics = item["metrics"]

        # Build condition name from config
        condition = config.get("name", "Unknown")
        if "collapse_level" in config:
            condition += f" (collapse={config['collapse_level']})"

        n_seeds = metrics["num_seeds"]

        acc = metrics["final_accuracy"]
        acc_str = f"{acc['mean']:.4f} ± {acc['std']:.4f}" if acc["mean"] is not None else "N/A"

        grok = metrics["grokking_step"]
        grok_str = f"{grok['mean']:.1f} ± {grok['std']:.1f}" if grok["mean"] is not None else "N/A"

        wn = metrics["weight_norm_change"]
        wn_str = f"{wn['mean']:.4f} ± {wn['std']:.4f}" if wn["mean"] is not None else "N/A"

        lines.append(f"| {condition} | {n_seeds} | {acc_str} | {grok_str} | {wn_str} |")

    return "\n".join(lines)


def find_best_condition(results: dict, metric: str = "final_accuracy") -> tuple[str, dict]:
    """Finds the best condition based on the given metric."""
    if not results:
        raise ValueError("Results dict is empty.")

    best_key = None
    best_val = None

    # Define optimization direction
    # final_accuracy -> higher is better
    # grokking_step -> lower is better (faster grokking)
    # weight_norm_change -> usually lower is better, but task doesn't specify. Assuming higher accuracy is best default.
    maximize = metric == "final_accuracy"

    for str_key, item in results.items():
        metrics = item["metrics"]

        if metric not in metrics:
            continue

        m_val = metrics[metric]["mean"]
        if m_val is None:
            continue

        if best_val is None or (maximize and m_val > best_val) or (not maximize and m_val < best_val):
            best_val = m_val
            best_key = str_key

    if best_key is None:
        raise ValueError(f"No valid data found for metric: {metric}")

    return best_key, results[best_key]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment Management Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-sweep
    parser_sweep = subparsers.add_parser("generate-sweep", help="Generate a parameter sweep")
    parser_sweep.add_argument("--base-config", required=True, type=str, help="JSON string of base config")
    parser_sweep.add_argument("--param-name", required=True, type=str, help="Parameter to sweep")
    parser_sweep.add_argument("--values", required=True, type=str, help="Comma-separated list of values")

    # aggregate
    parser_agg = subparsers.add_parser("aggregate", help="Aggregate experiment results")
    parser_agg.add_argument("--results-dir", required=True, type=str, help="Directory containing results.json files")

    # compare
    parser_comp = subparsers.add_parser("compare", help="Compare aggregated experiment results")
    parser_comp.add_argument("--results-dir", required=True, type=str, help="Directory containing results.json files")

    args = parser.parse_args()

    if args.command == "generate-sweep":
        base = json.loads(args.base_config)
        # Try to parse values as int/float if possible
        raw_vals = args.values.split(",")
        parsed_vals = []
        for v in raw_vals:
            try:
                parsed_vals.append(int(v))
            except ValueError:
                try:
                    parsed_vals.append(float(v))
                except ValueError:
                    parsed_vals.append(v)

        configs = generate_sweep(base, args.param_name, parsed_vals)
        for i, c in enumerate(configs):
            print(f"Config {i}: {json.dumps(c.to_dict())}")

    elif args.command == "aggregate":
        res = aggregate_results(args.results_dir)
        print(json.dumps(res, indent=2))

    elif args.command == "compare":
        res = aggregate_results(args.results_dir)
        print(compare_conditions(res))

if __name__ == "__main__":
    main()
