import re

with open("src/experiments/runner.py", "r") as f:
    content = f.read()

# Replace train_epoch with nothing
content = re.sub(r'    def train_epoch\(self\) -> dict:.*?        return \{\n            "loss": total_loss / total,\n            "accuracy": correct / total\n        \}\n', '', content, flags=re.DOTALL)

# Replace evaluate
eval_replacement = """    def _evaluate_loader(self, loader) -> tuple:
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
"""

content = re.sub(r'    def evaluate\(self\) -> dict:.*?        return \{\n            "loss": total_loss / total,\n            "accuracy": correct / total\n        \}\n', eval_replacement, content, flags=re.DOTALL)

# Replace the run loop
run_replacement = """        step = 0
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

            if self.config.collapse_config and self.config.collapse_config.injection_point == "optimizer" \\
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
        }"""

content = re.sub(r'        # Max steps translates roughly to epochs depending on batch size.*?            "history": history\n        \}', run_replacement, content, flags=re.DOTALL)

with open("src/experiments/runner.py", "w") as f:
    f.write(content)
