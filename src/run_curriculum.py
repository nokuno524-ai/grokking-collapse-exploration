import argparse
import os
from src.train import train, TrainConfig

def run_experiment_matrix(output_dir: str, max_steps: int, seeds: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    # Define experiment configurations
    experiments = []

    # 1. Constant baselines
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        experiments.append({
            "name": f"constant_w{w:.2f}",
            "schedule": "constant",
            "start_w": w,
            "end_w": w
        })

    # 2. Annealing schedules (Collapsed -> Pure)
    for sched in ["linear", "cosine", "step"]:
        experiments.append({
            "name": f"{sched}_collapse_to_pure",
            "schedule": sched,
            "start_w": 1.0,
            "end_w": 0.0
        })

    # 3. Annealing schedules (Pure -> Collapsed)
    for sched in ["linear", "cosine", "step"]:
        experiments.append({
            "name": f"{sched}_pure_to_collapse",
            "schedule": sched,
            "start_w": 0.0,
            "end_w": 1.0
        })

    for seed in range(42, 42 + seeds):
        for exp in experiments:
            condition_name = f"{exp['name']}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Running condition: {condition_name}")
            print(f"{'='*60}")

            config = TrainConfig(
                condition_name=condition_name,
                output_dir=output_dir,
                max_steps=max_steps,
                seed=seed,
                # Use a small severity for these experiments
                collapse_severity=0.5,
                curriculum_schedule=exp["schedule"],
                curriculum_start_w=exp["start_w"],
                curriculum_end_w=exp["end_w"],
                # Smaller model to keep CPU execution fast
                d_model=64,
                d_ff=256,
                n_heads=2,
                eval_every=500,
                log_every=500,
            )

            try:
                train(config)
            except Exception as e:
                print(f"Failed {condition_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/curriculum")
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()

    run_experiment_matrix(args.output_dir, args.max_steps, args.seeds)
