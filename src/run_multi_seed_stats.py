import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool
import logging

try:
    from .train import TrainConfig, train
    from .data import get_all_conditions
    from .analysis.grok_detector.detectors import detect_grokking_step
except ImportError:
    from train import TrainConfig, train
    from data import get_all_conditions
    from analysis.grok_detector.detectors import detect_grokking_step

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_seed(kwargs):
    """Worker function for running a single seed experiment."""
    seed = kwargs['seed']
    cond_name = kwargs['cond_name']
    cond_cfg = kwargs['cond_cfg']
    output_root = kwargs['output_root']
    max_steps = kwargs['max_steps']
    eval_every = kwargs['eval_every']

    # Small configuration for fast execution
    train_config = TrainConfig(
        prime=cond_cfg.prime,
        d_model=32,       # Small model
        n_heads=2,
        d_ff=128,
        n_layers=1,
        max_steps=max_steps,
        eval_every=eval_every,
        log_every=max_steps + 1,  # less logging
        save_every=max_steps + 1, # no checkpointing
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=cond_cfg.collapse_level,
        collapse_severity=cond_cfg.collapse_severity,
        seed=seed,
        condition_name=cond_name,
        output_dir=str(Path(output_root) / f"seed_{seed}"),
    )

    # Train the model
    state = train(train_config)

    # Extract accuracies over time for this seed
    steps = [entry['step'] for entry in state.history]
    test_accs = [entry['test_acc'] for entry in state.history]

    # Use our new detector
    import numpy as np
    grok_step, grok_band = detect_grokking_step(np.array(steps), np.array(test_accs))

    # Construct result dict
    result = {
        "seed": seed,
        "condition": cond_name,
        "collapse_level": cond_cfg.collapse_level,
        "final_test_acc": state.test_acc,
        "grokked": grok_step is not None,
        "grokking_step": grok_step,
        "grokking_band": grok_band,
        "history_steps": steps,
        "history_test_acc": test_accs
    }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds to run per condition")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Number of training steps per seed (small for CPU)")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Eval frequency")
    parser.add_argument("--output-file", type=str, default="results/multi_seed_stats.jsonl",
                        help="Path to output JSONL file")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    conditions = get_all_conditions()

    tasks = []
    for cond_name, cond_cfg in conditions.items():
        for seed in range(42, 42 + args.n_seeds):
            tasks.append({
                'seed': seed,
                'cond_name': cond_name,
                'cond_cfg': cond_cfg,
                'output_root': "results/temp_stats_runs",
                'max_steps': args.max_steps,
                'eval_every': args.eval_every,
            })

    logging.info(f"Generated {len(tasks)} tasks ({args.n_seeds} seeds x {len(conditions)} conditions)")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    with Pool(args.workers) as p:
        for i, result in enumerate(p.imap_unordered(run_seed, tasks)):
            logging.info(f"Completed {i+1}/{len(tasks)}: seed={result['seed']} condition={result['condition']} (Grokked: {result['grokked']}, Step: {result['grokking_step']})")
            with open(args.output_file, 'a') as f:
                f.write(json.dumps(result) + "\n")

    logging.info(f"Finished. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
