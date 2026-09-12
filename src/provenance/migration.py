import os
import json
import glob
import subprocess
from typing import Dict, Any

from src.provenance.registry import register, _load_registry
from src.provenance.lineage import generate_dataset_hashes, compute_config_hash

def get_git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def migrate_existing_experiments(results_dir: str = "results"):
    """
    Finds all results.json in the results_dir, reconstructs the registry entries.
    """
    git_sha = get_git_sha()

    # Use glob to find all results.json files recursively
    results_files = glob.glob(f"{results_dir}/**/results.json", recursive=True)

    failed_files = []
    registered_count = 0

    for rf in results_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)

            if "config" not in data:
                print(f"Skipping {rf} - no config found.")
                failed_files.append(rf)
                continue

            config = data["config"]

            # Reconstruct the hashes based on the saved config
            try:
                dataset_hashes = generate_dataset_hashes(config)
            except Exception as e:
                print(f"Skipping {rf} - dataset hash generation failed: {e}")
                failed_files.append(rf)
                continue

            config_hash = compute_config_hash(config)

            # Generate a unique run ID based on path or config hash
            # We'll use the relative path as a descriptive identifier
            rel_path = os.path.relpath(rf, results_dir)
            run_id = rel_path.replace('/results.json', '').replace('/', '_')
            if not run_id:
                run_id = f"root_{config_hash[:8]}"

            # If there's a naming collision, append a hash
            registry = _load_registry()
            if run_id in registry:
                run_id = f"{run_id}_{config_hash[:8]}"

            # Lineage tracking
            # Determine if this config uses tree_depth or is derived from another run
            lineage_info = {
                "seed": config.get("seed", 42),
                "collapse_level": config.get("collapse_level", 0.0),
                "collapse_severity": config.get("collapse_severity", 0.5),
                "tree_depth": config.get("tree_depth", 0)
            }

            entry = {
                "run_id": run_id,
                "git_sha": git_sha,
                "config_hash": config_hash,
                "dataset_hashes": dataset_hashes,
                "lineage": lineage_info,
                "metrics": {
                    "grokked": data.get("grokked"),
                    "grokking_step": data.get("grokking_step"),
                    "final_test_acc": data.get("final_test_acc"),
                    "final_train_acc": data.get("final_train_acc")
                },
                "artifacts": {
                    "results_file": rf,
                    "checkpoints_dir": os.path.dirname(rf)
                }
            }

            register(run_id, entry)
            registered_count += 1

        except Exception as e:
            print(f"Error migrating {rf}: {e}")
            failed_files.append(rf)

    print(f"Migration complete. Registered {registered_count} experiments.")
    if failed_files:
        print(f"Failed to migrate {len(failed_files)} files:")
        for f in failed_files:
            print(f" - {f}")

if __name__ == "__main__":
    migrate_existing_experiments()
