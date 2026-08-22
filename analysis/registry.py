import json
import hashlib
from pathlib import Path

def get_config_hash(config):
    # Create a deterministic hash of the config parameters
    # Ignore output_dir and condition_name as they are path/run specific
    stable_config = {k: v for k, v in config.items() if k not in ["output_dir", "condition_name"]}
    config_str = json.dumps(stable_config, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

def build_registry(results_dir: Path, output_file: Path):
    registry = []
    for p in results_dir.rglob("results.json"):
        # We don't want to include the registry itself if it gets named results.json
        if p.name == "registry.json":
            continue

        try:
            with open(p) as f:
                data = json.load(f)

            config = data.get("config", {})
            config_hash = get_config_hash(config)

            # Map condition logic based on config
            noise_fraction = config.get("noise_fraction", config.get("collapse_level", 0.0))
            wd = config.get("weight_decay", 1.0)

            entry = {
                "run_path": str(p.parent),
                "config_hash": config_hash,
                "seed": config.get("seed", 0),
                "weight_decay": wd,
                "noise_fraction": noise_fraction,
                "collapse_level": config.get("collapse_level", 0.0),
                "collapse_severity": config.get("collapse_severity", 0.0),
                "train_fraction": config.get("train_fraction", 0.3),
                "condition_name": config.get("condition_name", "unknown"),
                "grokked": data.get("grokked", False),
                "grokking_step": data.get("grokking_step"),
                "final_train_acc": data.get("final_train_acc"),
                "final_test_acc": data.get("final_test_acc"),
                "final_weight_norm": data.get("final_weight_norm"),
                "final_embedding_rank": data.get("final_embedding_rank"),
                "final_fourier_concentration": data.get("final_fourier_concentration")
            }
            registry.append(entry)
        except Exception as e:
            print(f"Error parsing {p}: {e}")

    with open(output_file, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"Registry built with {len(registry)} entries at {output_file}")
    return registry

if __name__ == "__main__":
    results_dir = Path("results")
    output_file = results_dir / "registry.json"
    build_registry(results_dir, output_file)
