import argparse
from pathlib import Path
from src.train import train, TrainConfig

def run_recovery(args: argparse.Namespace) -> None:
    """Contaminate early, then clean up and measure time to grok."""
    for seed in args.seeds:
        print(f"--- Recovery Run | switch_step={args.switch_step} | seed={seed} ---")
        config = TrainConfig(
            max_steps=args.max_steps,
            seed=seed,
            collapse_level=args.collapse_level,
            schedule_type="switch",
            switch_step=args.switch_step,
            switch_to=0.0,  # Recover to pure
            condition_name=f"recovery_S{args.switch_step}_seed{seed}",
            output_dir=args.output_dir
        )
        train(config)

def run_post_grok_poisoning(args: argparse.Namespace) -> None:
    """Train pure until grokking (or step S), then poison to see if grokking unravels."""
    for seed in args.seeds:
        print(f"--- Poison Run | switch_step={args.switch_step} | seed={seed} ---")
        config = TrainConfig(
            max_steps=args.max_steps,
            seed=seed,
            collapse_level=0.0, # Start pure
            schedule_type="switch",
            switch_step=args.switch_step,
            switch_to=args.collapse_level, # Poison
            condition_name=f"poison_S{args.switch_step}_seed{seed}",
            output_dir=args.output_dir
        )
        train(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["recovery", "poison"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--switch-step", type=int, default=25000)
    parser.add_argument("--collapse-level", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="results/schedules")
    args = parser.parse_args()

    if args.mode == "recovery":
        run_recovery(args)
    elif args.mode == "poison":
        run_post_grok_poisoning(args)
