import argparse
import sys
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from src.checkpoint import load_model_from_run, load_run
from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention_metrics
from src.attention_visualization import (
    plot_comparison_heatmaps,
    plot_evolution_frames,
    plot_metrics_over_time
)
import re

def get_available_steps(run_dir: Path) -> Set[int]:
    """Get set of available steps in a run directory."""
    steps = set()
    for p in run_dir.glob("checkpoint_*.pt"):
        match = re.findall(r"\d+", p.name)
        if match:
            steps.add(int(match[-1]))
    return steps

def main():
    parser = argparse.ArgumentParser(description="Compare attention patterns between pure and collapsed models.")
    parser.add_argument("--pure-dir", type=str, required=True, help="Directory containing pure model checkpoints")
    parser.add_argument("--col-dir", type=str, required=True, help="Directory containing collapsed model checkpoints")
    parser.add_argument("--out-dir", type=str, default="analysis/attention", help="Output directory for plots")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of eval samples to average over")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for eval samples")

    args = parser.parse_args()

    pure_dir = Path(args.pure_dir)
    col_dir = Path(args.col_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pure_steps = get_available_steps(pure_dir)
    col_steps = get_available_steps(col_dir)
    common_steps = sorted(list(pure_steps.intersection(col_steps)))

    if not common_steps:
        print("No common steps found between the two directories.")
        return

    print(f"Found {len(common_steps)} common steps. Processing...")

    _, pure_cfg = load_run(pure_dir, common_steps[-1])

    model_kwargs = {
        'prime': pure_cfg.get('prime', 59),
        'd_model': pure_cfg.get('d_model', 128),
        'n_heads': pure_cfg.get('n_heads', 4),
        'd_ff': pure_cfg.get('d_ff', 512),
        'n_layers': pure_cfg.get('n_layers', 1),
    }

    pure_model = ModularArithmeticTransformer(**model_kwargs)
    col_model = ModularArithmeticTransformer(**model_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pure_model = pure_model.to(device)
    col_model = col_model.to(device)

    torch.manual_seed(args.seed)
    eval_inputs = torch.randint(0, model_kwargs['prime'], (args.num_samples, 2), device=device)

    metrics_history = {
        'pure': {'entropy': {}, 'diag_ratio': {}, 'specialization': {}},
        'collapsed': {'entropy': {}, 'diag_ratio': {}, 'specialization': {}}
    }

    pure_weights_history = {}
    col_weights_history = {}

    for step in common_steps:
        print(f"Processing step {step}...")

        load_model_from_run(pure_model, pure_dir, step, strict=False)
        load_model_from_run(col_model, col_dir, step, strict=False)

        _, pure_metrics = extract_attention_metrics(pure_model, eval_inputs)
        _, col_metrics = extract_attention_metrics(col_model, eval_inputs)

        pure_weights = pure_metrics['weights'][0]
        col_weights = col_metrics['weights'][0]

        pure_weights_history[step] = pure_weights
        col_weights_history[step] = col_weights

        plot_comparison_heatmaps(
            pure_weights=pure_weights,
            collapsed_weights=col_weights,
            step=step,
            output_path=out_dir / f"compare_step_{step:06d}.png",
            layer_idx=0
        )

        for metric_name in ['entropy', 'diag_ratio', 'specialization']:
            metrics_history['pure'][metric_name][step] = pure_metrics[metric_name]
            metrics_history['collapsed'][metric_name][step] = col_metrics[metric_name]

    print("Generating evolution frames...")
    plot_evolution_frames(pure_weights_history, out_dir / "evolution_pure", prefix="pure", layer_idx=0)
    plot_evolution_frames(col_weights_history, out_dir / "evolution_collapsed", prefix="collapsed", layer_idx=0)

    print("Plotting scalar metrics over time...")
    plot_metrics_over_time(metrics_history, out_dir / "metrics", labels=['pure', 'collapsed'])

    print(f"Done. Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()