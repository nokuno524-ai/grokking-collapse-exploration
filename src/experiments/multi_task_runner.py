"""
Runner for multi-task experiments.
Tests the model on various tasks across different collapse levels and seeds.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Callable

from src.data import DatasetConfig
from src.data_multi import (
    generate_polynomial_arithmetic,
    generate_composition_task,
    generate_permutation_task,
    generate_sorting_task
)
from src.model import ModularArithmeticTransformer, TransformerConfig
from src.train import Trainer, TrainConfig

class MultiTaskRunner:
    """Runs multiple tasks across different collapse levels and seeds."""
    def __init__(self, tasks: Dict[str, Callable], collapse_levels: List[float], base_config: DatasetConfig):
        self.tasks = tasks
        self.collapse_levels = collapse_levels
        self.base_config = base_config

    def _get_dataset(self, task_name: str, config: DatasetConfig):
        if task_name == "polynomial":
            return generate_polynomial_arithmetic(config)
        elif task_name == "composition":
            return generate_composition_task(config)
        elif task_name == "permutation":
            return generate_permutation_task(config)
        elif task_name == "sorting":
            return generate_sorting_task(config)
        else:
            raise ValueError(f"Unknown task {task_name}")

    def run_single(self, task_name: str, collapse_level: float, seed: int) -> dict:
        """Run a single experiment for a task, collapse level, and seed."""
        # Create dataset config
        config = DatasetConfig(
            prime=self.base_config.prime,
            train_fraction=self.base_config.train_fraction,
            collapse_level=collapse_level,
            collapse_severity=self.base_config.collapse_severity,
            noise_fraction=self.base_config.noise_fraction,
            seed=seed
        )

        # Generate data
        train_in, train_tgt, test_in, test_tgt = self._get_dataset(task_name, config)

        # Vocab size heuristic based on data and task
        vocab_size = max(train_in.max().item(), test_in.max().item(), train_tgt.max().item(), test_tgt.max().item()) + 1
        seq_len = train_in.shape[1]

        # Model config
        model_config = TransformerConfig(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=128,
            n_heads=4,
            d_ff=512,
            n_layers=1,
            output_dim=vocab_size
        )

        model = ModularArithmeticTransformer(model_config)

        # Train config
        train_config = TrainConfig(
            max_steps=1000, # Short max steps for quick multi-task testing
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1.0,
            eval_every=100
        )

        # For these tests, mock an output dir if not saving
        output_dir = f"results/multi_task/{task_name}/collapse_{collapse_level}/seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)

        trainer = Trainer(
            model=model,
            train_inputs=train_in,
            train_targets=train_tgt,
            test_inputs=test_in,
            test_targets=test_tgt,
            config=train_config,
            output_dir=output_dir,
            use_wandb=False
        )

        history = trainer.train()

        # Just extract key final metrics or "did it grok"
        final_test_acc = history["test_acc"][-1]
        final_train_acc = history["train_acc"][-1]

        # Simple grokking definition: train acc > 0.99, test acc > 0.90
        grokking = (final_train_acc > 0.99) and (final_test_acc > 0.90)

        results = {
            "task": task_name,
            "collapse_level": collapse_level,
            "seed": seed,
            "final_test_acc": final_test_acc,
            "final_train_acc": final_train_acc,
            "grokking": grokking,
            "history": history
        }

        # Save results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_all(self, seed: int = 42) -> dict:
        """Run all tasks across all collapse levels for a single seed."""
        results = {}
        for task_name in self.tasks.keys():
            for collapse in self.collapse_levels:
                key = f"{task_name}:{collapse}"
                print(f"Running {key} with seed {seed}...")
                res = self.run_single(task_name, collapse, seed)
                results[key] = res
        return results

    def run_with_seeds(self, n_seeds: int) -> dict:
        """Run experiments across multiple seeds."""
        all_results = {}
        for task_name in self.tasks.keys():
            all_results[task_name] = {}
            for collapse in self.collapse_levels:
                all_results[task_name][collapse] = []
                for s in range(n_seeds):
                    seed = self.base_config.seed + s
                    print(f"Running {task_name} | collapse: {collapse} | seed: {seed}")
                    res = self.run_single(task_name, collapse, seed)
                    all_results[task_name][collapse].append(res)
        return all_results
