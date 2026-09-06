import torch
import numpy as np
import json
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, List

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def extract_dataset_hidden_states(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract hidden states for the entire dataset."""
    model.eval()
    all_h = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_in = inputs[i:i+batch_size].to(device)

            tok = model.token_embed(batch_in)
            positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_in.shape[0], -1)
            pos = model.pos_embed(positions)
            h = tok + pos
            h = model.transformer(h)
            h = model.ln(h)
            h = h.mean(dim=1)

            all_h.append(h.cpu().numpy())

    return np.vstack(all_h), inputs.numpy(), targets.numpy()

def train_probes(
    hidden_states: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    prime: int
) -> Dict[str, float]:
    """
    Train linear probes on frozen hidden states and return accuracies.
    Probes:
    - parity_a: Is operand a even?
    - parity_b: Is operand b even?
    - result_bucket: Is (a+b)%p < p/2?
    """
    labels_parity_a = (inputs[:, 0] % 2 == 0).astype(int)
    labels_parity_b = (inputs[:, 1] % 2 == 0).astype(int)
    labels_result = (targets < prime / 2).astype(int)

    accuracies = {}

    for name, y in [
        ("parity_a", labels_parity_a),
        ("parity_b", labels_parity_b),
        ("result_bucket", labels_result)
    ]:
        # Using a simple Logistic Regression
        clf = LogisticRegression(max_iter=1000)
        try:
            clf.fit(hidden_states, y)
            preds = clf.predict(hidden_states)
            accuracies[name] = accuracy_score(y, preds)
        except Exception:
            # Handle cases where all labels might be same (unlikely for full dataset but safe)
            accuracies[name] = 0.5

    return accuracies

def run_probe_tracker(run_dir: Path, device: torch.device) -> None:
    """Iterate over checkpoints, run probes, log to JSONL."""
    run_dir = Path(run_dir)
    if not (run_dir / "results.json").exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")

    with open(run_dir / "results.json") as f:
        config_dict = json.load(f)["config"]
    config = DatasetConfig(**config_dict)

    # We probe on the test set for generalization (or whole set)
    _, _, test_in, test_tgt = generate_modular_arithmetic(config)

    out_file = run_dir / "hidden_state_probes.jsonl"
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

    with open(out_file, "w") as f:
        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            step = ckpt["step"]

            model = ModularArithmeticTransformer(prime=config.prime).to(device)
            model.load_state_dict(ckpt["model_state"])

            h, x, y = extract_dataset_hidden_states(model, test_in, test_tgt, device)
            accs = train_probes(h, x, y, config.prime)

            log_obj = {"step": step, "accuracies": accs}
            f.write(json.dumps(log_obj) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_probe_tracker(Path(args.run_dir), device)
