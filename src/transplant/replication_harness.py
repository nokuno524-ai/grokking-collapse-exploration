import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict

from src.transplant.run_transplants import run_transplant_experiment, VariantResult
from src.transplant.stats import cohens_d, bootstrap_ci, check_replication

def run_replication(
    seeds: List[int],
    conditions: List[str],
    results_dir: Path,
    output_dir: Path,
    components: List[str],
    rescue_steps: int = 0
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    # Run transplants across seeds and conditions
    for cond in conditions:
        print(f"\n=============================================")
        print(f"Running condition: {cond}")
        print(f"=============================================")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            pure_run = results_dir / str(seed) / "pure"
            contam_run = results_dir / str(seed) / cond

            if not pure_run.exists() or not contam_run.exists():
                print(f"[warn] Missing runs for seed {seed}, condition {cond}. Skipping.")
                continue

            out_subdir = output_dir / cond / str(seed)
            try:
                results = run_transplant_experiment(
                    pure_run=pure_run,
                    contam_run=contam_run,
                    output_dir=out_subdir,
                    components=components,
                    rescue_steps=rescue_steps,
                    seed=seed,
                )

                for r in results:
                    r_dict = asdict(r)
                    r_dict["seed"] = seed
                    r_dict["condition"] = cond
                    all_results.append(r_dict)
            except Exception as e:
                print(f"[error] Failed to run transplant for seed {seed}, condition {cond}: {e}")

    # Aggregate and compute stats
    stats_results = []

    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        if not cond_results:
            continue

        for comp in components:
            # We compare baseline_contam to transplant_{comp} for zero-shot accuracy
            baseline_accs = []
            transplant_accs = []

            for seed in seeds:
                b_run = next((r for r in cond_results if r["seed"] == seed and r["name"] == "baseline_contam"), None)
                t_run = next((r for r in cond_results if r["seed"] == seed and r["name"] == f"transplant_{comp}"), None)

                if b_run and t_run:
                    baseline_accs.append(b_run["test_acc"])
                    transplant_accs.append(t_run["test_acc"])

            if not baseline_accs:
                continue

            # Effect size (Cohen's d, paired)
            d = cohens_d(baseline_accs, transplant_accs, paired=True)

            # Differences
            diffs = [t - b for b, t in zip(baseline_accs, transplant_accs)]
            lower, upper = bootstrap_ci(diffs)
            mean_diff = sum(diffs) / len(diffs)

            replicates = check_replication(diffs)

            stats_results.append({
                "condition": cond,
                "component": comp,
                "mean_baseline_acc": sum(baseline_accs)/len(baseline_accs),
                "mean_transplant_acc": sum(transplant_accs)/len(transplant_accs),
                "mean_diff": mean_diff,
                "ci_lower": lower,
                "ci_upper": upper,
                "cohens_d": d,
                "replicates": replicates,
                "n_seeds": len(baseline_accs),
                "raw_diffs": diffs,
            })

    # Save consolidated results
    with open(output_dir / "replication_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "replication_stats.json", "w") as f:
        json.dump(stats_results, f, indent=2)

    # Generate markdown table
    md = "# Circuit Transplant Replication\n\n"
    md += "| Condition | Component | N | Baseline Acc | Transplant Acc | Mean Diff (95% CI) | Cohen's d | Replicates |\n"
    md += "|-----------|-----------|---|--------------|----------------|--------------------|-----------|------------|\n"

    for r in stats_results:
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        rep_str = "✅ Yes" if r["replicates"] else "❌ No"
        md += f"| {r['condition']} | {r['component']} | {r['n_seeds']} | {r['mean_baseline_acc']:.3f} | {r['mean_transplant_acc']:.3f} | {r['mean_diff']:.3f} {ci_str} | {r['cohens_d']:.3f} | {rep_str} |\n"

    with open(output_dir / "replication_summary.md", "w") as f:
        f.write(md)

    print(f"\n[info] Replication finished. Results saved to {output_dir}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--conditions", type=str, default="low_collapse,medium_collapse,high_collapse,severe_collapse")
    ap.add_argument("--results-dir", type=Path, default=Path("results/multi_seed"))
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/transplant_replication"))
    ap.add_argument("--components", type=str, default="token_embed,self_attn_in_proj,self_attn_out_proj,linear1,linear2,output_head")
    ap.add_argument("--rescue-steps", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="Run a fast smoke test with minimal steps.")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    components = [c.strip() for c in args.components.split(",")]

    if args.smoke:
        print("[info] Running in smoke test mode...")
        # Override to just one quick thing if we have smoke tests, but for now we'll just run normally on whatever we have

    run_replication(
        seeds=seeds,
        conditions=conditions,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        components=components,
        rescue_steps=args.rescue_steps
    )

if __name__ == "__main__":
    main()
