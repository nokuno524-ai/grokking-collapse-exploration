import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from src.model import ModularArithmeticTransformer
import json
import matplotlib.pyplot as plt

def collect_hidden(checkpoint_path: Path, probe_batch: torch.Tensor, is_test: bool = False, test_model=None) -> Dict[str, torch.Tensor]:
    """
    Load a model from a checkpoint and collect hidden states for a given batch.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.
        probe_batch: Input tensor of shape (batch, 2).
        is_test: If True, uses the provided test_model instead of dynamic loading.
        test_model: Dummy model for pytest.

    Returns:
        A dictionary mapping layer names to extracted CPU tensors of shape (batch, d_model).
    """
    d = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if is_test:
        model = test_model
    else:
        # Get config to instantiate model dynamically
        config = d.get('config', {})
        # Extract model arguments from config
        model_kwargs = {}
        for k in ['prime', 'd_model', 'n_heads', 'd_ff', 'n_layers', 'dropout']:
            if k in config:
                model_kwargs[k] = config[k]
        model = ModularArithmeticTransformer(**model_kwargs)

    state_dict = d.get('model_state', d.get('model_state_dict'))
    if state_dict is None:
        raise ValueError(f"Could not find model state in {checkpoint_path}")

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    hidden_states = {}

    if is_test:
        # Standard hook for the dummy test model
        def get_activation(name):
            def hook(model, input, output):
                hidden_states[name] = output.detach()
            return hook
        handles = []
        handles.append(model.layer1.register_forward_hook(get_activation('layer1')))
        handles.append(model.layer2.register_forward_hook(get_activation('layer2')))
        with torch.no_grad():
            model(probe_batch)
        for handle in handles:
            handle.remove()
        return hidden_states

    # Register hooks for ModularArithmeticTransformer
    def get_activation(name):
        def hook(model, input, output):
            hidden_states[name] = output.mean(dim=1).detach()
        return hook

    handles = []
    handles.append(model.transformer.register_forward_hook(get_activation('transformer')))
    handles.append(model.ln.register_forward_hook(get_activation('ln')))

    with torch.no_grad():
        batch_size = probe_batch.shape[0]
        tok = model.token_embed(probe_batch)
        positions = torch.arange(2, device=probe_batch.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        embed_out = tok + pos
        hidden_states['embed'] = embed_out.mean(dim=1).detach()

        model(probe_batch)

    for handle in handles:
        handle.remove()

    return hidden_states

def train_linear_probe(X: np.ndarray, y: np.ndarray, k_fold: int = 5) -> Tuple[float, float]:
    """
    Train a linear probe using logistic regression with k-fold cross-validation.
    """
    if len(X) < k_fold:
        k_fold = len(X)

    if k_fold < 2:
        return np.nan, np.nan

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    accs = []

    if len(np.unique(y)) <= 1:
        return np.nan, np.nan

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) <= 1:
            accs.append(np.nan)
            continue

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accs.append(acc)

    valid_accs = [a for a in accs if not np.isnan(a)]
    if not valid_accs:
        return np.nan, np.nan

    return float(np.mean(valid_accs)), float(np.std(valid_accs))

def get_run_info(results_json_path: Path, threshold: float = 0.95) -> Tuple[bool, float]:
    """
    Determine if a run grokked and extract its collapse severity.
    """
    with open(results_json_path, 'r') as f:
        data = json.load(f)

    grokked = data.get('grokked', False)
    if 'grokked' not in data and 'history' in data:
        for step_data in data['history']:
            if step_data.get('test_acc', 0.0) >= threshold:
                grokked = True
                break

    severity = 0.0
    if 'config' in data:
        severity = data['config'].get('collapse_severity', 0.0)

    return grokked, severity

def probe_separation_curve(checkpoint_dir: Path, output_dir: Path = Path('analysis')):
    """
    Generate probe separation curve per severity across training steps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    p = 59
    all_pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])

    runs = []
    for results_file in checkpoint_dir.rglob("results.json"):
        run_dir = results_file.parent
        label, severity = get_run_info(results_file)
        runs.append((run_dir, label, severity))

    if not runs:
        print(f"No runs found in {checkpoint_dir}")
        return

    steps = set()
    severities = set()
    for run_dir, _, severity in runs:
        severities.add(severity)
        for pt_file in run_dir.glob("checkpoint_*.pt"):
            step_str = pt_file.stem.split('_')[1]
            steps.add(int(step_str))

    steps = sorted(list(steps))
    severities = sorted(list(severities))

    print(f"Found steps: {steps}")
    print(f"Found severities: {severities}")
    print(f"Found {len(runs)} runs")

    layers = ['embed', 'transformer', 'ln']
    results = {sev: {layer: {'mean': [], 'std': [], 'steps': []} for layer in layers} for sev in severities}

    for sev in severities:
        for step in steps:
            X_dict = {layer: [] for layer in layers}
            y = []

            for run_dir, label, run_sev in runs:
                if run_sev != sev: continue

                ckpt_path = run_dir / f"checkpoint_{step}.pt"
                if ckpt_path.exists():
                    try:
                        hidden = collect_hidden(ckpt_path, all_pairs)
                        for layer in layers:
                            mean_activation = hidden[layer].mean(dim=0).numpy()
                            X_dict[layer].append(mean_activation)
                        y.append(1 if label else 0)
                    except Exception as e:
                        print(f"Error processing {ckpt_path}: {e}")

            if len(y) > 0:
                y = np.array(y)
                if len(np.unique(y)) > 1:
                    for layer in layers:
                        X = np.array(X_dict[layer])
                        mean_acc, std_acc = train_linear_probe(X, y)
                        results[sev][layer]['mean'].append(mean_acc)
                        results[sev][layer]['std'].append(std_acc)
                        results[sev][layer]['steps'].append(step)
                else:
                    print(f"Severity {sev} Step {step} only has one class. Skipping probe.")

    # Plotting
    num_severities = len(severities)
    fig, axes = plt.subplots(1, num_severities, figsize=(6 * num_severities, 6), sharey=True)
    if num_severities == 1:
        axes = [axes]

    for i, sev in enumerate(severities):
        ax = axes[i]
        for layer in layers:
            valid_steps = results[sev][layer]['steps']
            if valid_steps:
                means = np.array(results[sev][layer]['mean'])
                stds = np.array(results[sev][layer]['std'])
                ax.plot(valid_steps, means, label=layer, marker='o')
                ax.fill_between(valid_steps, means - stds, means + stds, alpha=0.2)

        ax.axhline(0.5, color='gray', linestyle='--', label='Chance (0.5)')
        ax.set_xlabel('Training Step')
        if i == 0:
            ax.set_ylabel('Probe Accuracy (Grokked vs Non-Grokked)')
        ax.set_title(f'Severity {sev}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Linear Probe Separation Curve (Per Severity)')
    out_file = output_dir / "probe_separation_curve.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_file}")
