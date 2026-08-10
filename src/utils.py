import os
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

def save_results(config, state, output_dir: Path):
    """Utility to save experiment results to a standard JSON format."""
    out_path = output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "config": asdict(config),
        "grokked": state.grokked,
        "grokking_step": state.grokking_step,
        "final_train_acc": state.train_acc,
        "final_test_acc": state.test_acc,
        "final_weight_norm": state.weight_norm,
        "final_embedding_rank": state.embedding_rank,
        "final_fourier_concentration": state.fourier_concentration,
        "history": state.history,
    }

    results_path = out_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Grokked: {state.grokked} at step {state.grokking_step}")
