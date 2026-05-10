"""
Prime-brittleness check.

Question: does the cliff observed at p=59 also appear at a different prime
(p=97, p=113), or is the exact location an artifact of p=59?

Sweep (defaults): p ∈ {59, 97, 113} × wd=1.0 × noise ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3}
                  × 5 seeds  =  3 · 6 · 5  =  90 runs.

Note: the model is reused as-is (d_model=128 etc.); only the prime / output_head
size changes. The default train_fraction=0.3 is held fixed across primes —
which means the *number* of training examples grows with p² · 0.3, so we are
keeping the relative split constant. (Holding absolute size fixed is a separate
ablation that requires changing train_fraction per prime; not done here.)

Layout: results/prime_sweep/p<P>/wd<W>/noise<N>/seed_<S>/results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

try:
    from .train import TrainConfig, train
except ImportError:
    from train import TrainConfig, train  # type: ignore


DEFAULT_PRIMES = [59, 97, 113]
DEFAULT_WDS = [1.0]
DEFAULT_NOISES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def fmt_p(p: int) -> str: return f"p{p}"
def fmt_wd(wd: float) -> str: return f"wd{wd:g}"
def fmt_noise(n: float) -> str: return f"noise{n:g}"


def build_tasks(primes, wds, noises, seeds) -> List[Tuple[int, float, float, int]]:
    return [(p, w, n, s) for p in primes for w in wds for n in noises for s in seeds]


def run_one(prime: int, wd: float, noise: float, seed: int,
            output_root: str, max_steps: int):
    out_dir = Path(output_root) / fmt_p(prime) / fmt_wd(wd) / fmt_noise(noise)
    cfg = TrainConfig(
        prime=prime,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=wd,
        collapse_level=0.0,
        collapse_severity=0.5,
        noise_fraction=noise,
        seed=seed,
        condition_name=f"seed_{seed}",
        output_dir=str(out_dir),
        max_steps=max_steps,
    )
    return train(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=str,
                    default=",".join(str(p) for p in DEFAULT_PRIMES))
    ap.add_argument("--weight-decays", type=str,
                    default=",".join(f"{x:g}" for x in DEFAULT_WDS))
    ap.add_argument("--noise-fractions", type=str,
                    default=",".join(f"{x:g}" for x in DEFAULT_NOISES))
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--output-dir", type=str, default="results/prime_sweep")
    ap.add_argument("--array-id", type=int, default=None)
    ap.add_argument("--print-task-count", action="store_true")
    args = ap.parse_args()

    primes = [int(x) for x in args.primes.split(",") if x]
    wds = [float(x) for x in args.weight_decays.split(",") if x]
    noises = [float(x) for x in args.noise_fractions.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    tasks = build_tasks(primes, wds, noises, seeds)

    if args.print_task_count:
        print(len(tasks))
        return
    if args.array_id is not None:
        if not 0 <= args.array_id < len(tasks):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        p, wd, n, s = tasks[args.array_id]
        print(f"[array {args.array_id}/{len(tasks)}] p={p} wd={wd} noise={n} seed={s}")
        run_one(p, wd, n, s, args.output_dir, args.max_steps)
        return

    for i, (p, wd, n, s) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] p={p} wd={wd} noise={n} seed={s}")
        run_one(p, wd, n, s, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
