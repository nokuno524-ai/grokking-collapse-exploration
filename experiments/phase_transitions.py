import argparse
from pathlib import Path
import json

try:
    from src.train import TrainConfig, train
except ImportError:
    from src.train import TrainConfig, train  # type: ignore

def fmt_cond(level: float, severity: float, noise: float, wd: float) -> str:
    return f"level{level:g}_sev{severity:g}_noise{noise:g}_wd{wd:g}"

def run_one(level: float, severity: float, noise: float, wd: float, seed: int,
            output_root: str, max_steps: int):
    cond_dir = fmt_cond(level, severity, noise, wd)
    out_dir = Path(output_root) / cond_dir / f"seed_{seed}"
    train_config = TrainConfig(
        prime=59,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=wd,
        collapse_level=level,
        collapse_severity=severity,
        noise_fraction=noise,
        seed=seed,
        condition_name=f"seed_{seed}",
        output_dir=str(Path(output_root) / cond_dir),
        max_steps=max_steps,
    )
    return train(train_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collapse-levels", type=str, default="0.0,0.25,0.5")
    parser.add_argument("--collapse-severities", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--label-noises", type=str, default="0.0,0.15,0.3")
    parser.add_argument("--weight-decays", type=str, default="0.0,1.0,3.0")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/phase_transitions")
    parser.add_argument("--array-id", type=int, default=None)
    args = parser.parse_args()

    levels = [float(x) for x in args.collapse_levels.split(",") if x.strip() != ""]
    severities = [float(x) for x in args.collapse_severities.split(",") if x.strip() != ""]
    noises = [float(x) for x in args.label_noises.split(",") if x.strip() != ""]
    wds = [float(x) for x in args.weight_decays.split(",") if x.strip() != ""]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    tasks = []
    for l in levels:
        for s in severities:
            for n in noises:
                for w in wds:
                    for seed in seeds:
                        tasks.append((l, s, n, w, seed))

    if args.array_id is not None:
        if not (0 <= args.array_id < len(tasks)):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        l, s, n, w, seed = tasks[args.array_id]
        print(f"[array {args.array_id}] level={l} severity={s} noise={n} wd={w} seed={seed}")
        run_one(l, s, n, w, seed, args.output_dir, args.max_steps)
        return

    for i, (l, s, n, w, seed) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] level={l} sev={s} noise={n} wd={w} seed={seed}\n{'='*60}")
        run_one(l, s, n, w, seed, args.output_dir, args.max_steps)

if __name__ == "__main__":
    main()
