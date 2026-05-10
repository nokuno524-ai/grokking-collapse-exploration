"""
Build a table of final mechanistic metrics across (ratio, seed) for the
real-LM contamination training results in results/contamination/.

Reads every ratio_<R>_seed_<S>.json under results/contamination/ and
writes:
  - results/contamination/summary_table.csv     (one row per run, final step)
  - results/contamination/summary_aggregate.csv (mean +- std grouped by ratio)
  - stdout: a printed Markdown table for paste into reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

DEFAULT_DIR = Path("/scratch/qzp4ta/grokking-collapse/results/contamination")
FNAME_RE = re.compile(r"^ratio_(\d+)_seed_(\d+)\.json$")

METRIC_FIELDS = [
    "perplexity",
    "train_loss",
    "attn_effective_rank",
    "repr_entropy",
    "cos_sim_mean",
    "cos_sim_std",
    "distinct_2",
    "distinct_3",
    "distinct_4",
]


def load_runs(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(root.glob("ratio_*_seed_*.json")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        ratio = int(m.group(1))
        seed = int(m.group(2))
        data = json.loads(path.read_text())
        history = data.get("history", [])
        if not history:
            continue
        final = history[-1]
        row = {
            "ratio_pct": ratio,
            "seed": seed,
            "final_step": final.get("step"),
            "n_logged": len(history),
            "weight_decay": data.get("weight_decay"),
            "lr": data.get("lr"),
            "max_steps": data.get("max_steps"),
        }
        for k in METRIC_FIELDS:
            row[k] = final.get(k)
        rows.append(row)
    return rows


def aggregate(rows: List[Dict]) -> List[Dict]:
    by_ratio: Dict[int, List[Dict]] = {}
    for r in rows:
        by_ratio.setdefault(r["ratio_pct"], []).append(r)
    out: List[Dict] = []
    for ratio in sorted(by_ratio):
        runs = by_ratio[ratio]
        agg = {"ratio_pct": ratio, "n_seeds": len(runs)}
        for k in METRIC_FIELDS:
            vals = [r[k] for r in runs if r.get(k) is not None]
            if not vals:
                agg[f"{k}_mean"] = None
                agg[f"{k}_std"] = None
                continue
            agg[f"{k}_mean"] = statistics.mean(vals)
            agg[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def write_csv(path: Path, rows: List[Dict], fields: List[str]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def print_markdown(agg: List[Dict]):
    cols = ["ratio_pct", "n_seeds"]
    for k in ["perplexity", "attn_effective_rank", "repr_entropy", "cos_sim_mean", "distinct_3"]:
        cols.append(f"{k}_mean")
        cols.append(f"{k}_std")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in agg:
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    rows = load_runs(args.root)
    print(f"# Loaded {len(rows)} run(s) from {args.root}")
    if not rows:
        return
    agg = aggregate(rows)

    per_run_fields = ["ratio_pct", "seed", "final_step", "n_logged",
                      "weight_decay", "lr", "max_steps"] + METRIC_FIELDS
    agg_fields = ["ratio_pct", "n_seeds"]
    for k in METRIC_FIELDS:
        agg_fields.append(f"{k}_mean")
        agg_fields.append(f"{k}_std")

    write_csv(args.root / "summary_table.csv", rows, per_run_fields)
    write_csv(args.root / "summary_aggregate.csv", agg, agg_fields)

    print(f"# Wrote {args.root/'summary_table.csv'}")
    print(f"# Wrote {args.root/'summary_aggregate.csv'}")
    print()
    print_markdown(agg)


if __name__ == "__main__":
    main()
