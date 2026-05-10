import os
import time
import json
from pathlib import Path
from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.experiments.config import ExperimentConfig, CollapseConfig
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from src.train import compute_fourier_concentration, seed_worker

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        import random
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Determine dataset config from experiment config
        severity_mapping = {"none": 0.0, "low": 0.3, "medium": 0.5, "high": 0.7, "severe": 0.9}
        severity_val = severity_mapping.get(config.collapse_config.severity, 0.5) if config.collapse_config else 0.5

        self.data_config = DatasetConfig(
            collapse_level=config.collapse_level,
            collapse_severity=severity_val,
            seed=config.seed
        )

        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(self.data_config)
        self.train_dataset = TensorDataset(train_in, train_tgt)
        self.test_dataset = TensorDataset(test_in, test_tgt)

        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.seed)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=loader_generator
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=False,
            worker_init_fn=seed_worker
        )

        # Setup Model
        self.model = ModularArithmeticTransformer().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def inject_collapse(self, model: nn.Module, config: CollapseConfig):
        """Apply collapse mechanism to the model or optimizer based on config."""
        if config.injection_point == "model" and config.collapse_type == "weight_noise":
            noise_scale = {"none": 0.0, "low": 0.01, "medium": 0.05, "severe": 0.1}.get(config.severity, 0.0)
            if noise_scale > 0:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * noise_scale)
        # Note: Data collapse is handled during data generation via DatasetConfig.
        # Gradient noise would be added in the training loop.


    def _evaluate_loader(self, loader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item() * inputs.shape[0]
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += inputs.shape[0]
        return total_loss / total, correct / total

    def evaluate(self) -> dict:
        loss, acc = self._evaluate_loader(self.test_loader)
        return {"loss": loss, "accuracy": acc}

    def run(self) -> dict:
        """Full training run with periodic logging."""
        history = []
        grokked = False
        grokking_step = None
        grokking_threshold = 0.95

        start_time = time.time()

        step = 0
        max_steps = self.config.epochs
        dataloader_iter = iter(self.train_loader)

        while step < max_steps:
            self.model.train()
            try:
                inputs, targets = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.train_loader)
                inputs, targets = next(dataloader_iter)
                if self.config.collapse_config and self.config.collapse_config.injection_point == "model":
                    self.inject_collapse(self.model, self.config.collapse_config)

            step += 1
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.collapse_config and self.config.collapse_config.injection_point == "optimizer" \
                    and self.config.collapse_config.collapse_type == "gradient_noise":
                noise_scale = {"none": 0.0, "low": 0.01, "medium": 0.05, "severe": 0.1}.get(self.config.collapse_config.severity, 0.0)
                if noise_scale > 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.add_(torch.randn_like(param.grad) * noise_scale)

            self.optimizer.step()

            if step % self.config.log_interval == 0 or step == max_steps:
                train_loss, train_acc = self._evaluate_loader(self.train_loader)
                test_loss, test_acc = self._evaluate_loader(self.test_loader)

                weight_norm = self.model.get_weight_norm()
                embedding_rank = self.model.get_embedding_rank()
                fourier_concentration = compute_fourier_concentration(self.model)

                if test_acc >= grokking_threshold and not grokked:
                    grokked = True
                    grokking_step = step

                entry = {
                    "step": step,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "weight_norm": weight_norm,
                    "embedding_rank": embedding_rank,
                    "fourier_concentration": fourier_concentration,
                }
                history.append(entry)

        results = {
            "config": asdict(self.config),
            "grokked": grokked,
            "grokking_step": grokking_step,
            "final_train_acc": train_acc if 'train_acc' in locals() else 0.0,
            "final_test_acc": test_acc if 'test_acc' in locals() else self._evaluate_loader(self.test_loader)[1],
            "final_weight_norm": self.model.get_weight_norm(),
            "final_embedding_rank": self.model.get_embedding_rank(),
            "final_fourier_concentration": compute_fourier_concentration(self.model),
            "history": history
        }

        if self.config.output_dir:
            self.save_results(results, self.config.output_dir)

        return results

    def save_results(self, results: dict, output_dir: str):
        """Save results and configuration to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
