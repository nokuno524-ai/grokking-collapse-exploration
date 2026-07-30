import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

def parse_results_json(filepath: Path) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        return json.load(f)

def aggregate_results(results_dir: str = "results", skip_dirs: Optional[List[str]] = None) -> pd.DataFrame:
    if skip_dirs is None:
        skip_dirs = ["grid", "multi_seed", "seed_sweep", "noise_baseline", "scarcity_baseline", "exp_c_grid", "contamination"]

    base_path = Path(results_dir)
    data = []

    for json_path in base_path.rglob("results.json"):
        # Check if it should be skipped
        skip = False
        for parent in json_path.parents:
            if parent.name in skip_dirs:
                skip = True
                break
        if skip:
            continue

        res = parse_results_json(json_path)
        config = res.get("config", {})

        # Determine condition name from path if not in config
        condition = config.get("condition_name")
        if condition is None:
            # Get the parent folder that is one level below results
            for p in json_path.parents:
                if p.parent == base_path:
                    condition = p.name
                    break

        row = {
            "condition": condition,
            "seed": config.get("seed"),
            "collapse_level": config.get("collapse_level"),
            "collapse_severity": config.get("collapse_severity"),
            "grokked": res.get("grokked", False),
            "grokking_step": res.get("grokking_step"),
            "final_train_acc": res.get("final_train_acc"),
            "final_test_acc": res.get("final_test_acc"),
            "final_weight_norm": res.get("final_weight_norm"),
            "final_embedding_rank": res.get("final_embedding_rank"),
            "attention_entropy": res.get("attention_entropy", None),
            "circuit_importance": res.get("circuit_importance", None),
            "final_fourier_concentration": res.get("final_fourier_concentration"),
            "history": res.get("history", []),
        }
        data.append(row)

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = aggregate_results()
    print(df.drop(columns=["history"]).to_string())
