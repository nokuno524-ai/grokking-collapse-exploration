import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm


def collect_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for res_file in results_dir.glob("*/results.json"):
        with open(res_file) as f:
            data = json.load(f)
        cfg = data["config"]
        # Handle right censoring
        max_step = cfg["max_steps"]
        grokked = data["grokked"]
        grok_step = data["grokking_step"] if grokked else max_step

        # Difficulty proxy metric: prime ^ num_operands
        # Since input space is prime^num_operands
        difficulty_score = cfg["prime"] ** cfg["num_operands"]

        rows.append({
            "condition": cfg["condition_name"],
            "type": "capacity" if "capacity" in cfg["condition_name"] else "difficulty",
            "prime": cfg["prime"],
            "num_operands": cfg["num_operands"],
            "difficulty_score": difficulty_score,
            "collapse_level": cfg["collapse_level"],
            "seed": cfg["seed"],
            "d_model": cfg["d_model"],
            "grokked": grokked,
            "grok_step": grok_step,
            "max_step": max_step
        })
    return pd.DataFrame(rows)


def analyze_interaction(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No results found.")
        return

    diff_df = df[df["type"] == "difficulty"]
    if diff_df.empty:
        return

    # Standardize inputs for OLS to make coefficients comparable
    X = diff_df[["difficulty_score", "collapse_level"]].copy()

    diff_std = X["difficulty_score"].std()
    if pd.isna(diff_std) or diff_std == 0:
        X["difficulty_score"] = X["difficulty_score"] - X["difficulty_score"].mean()
    else:
        X["difficulty_score"] = (X["difficulty_score"] - X["difficulty_score"].mean()) / diff_std

    col_std = X["collapse_level"].std()
    if pd.isna(col_std) or col_std == 0:
        X["collapse_level"] = X["collapse_level"] - X["collapse_level"].mean()
    else:
        X["collapse_level"] = (X["collapse_level"] - X["collapse_level"].mean()) / col_std

    X["interaction"] = X["difficulty_score"] * X["collapse_level"]
    X = sm.add_constant(X)
    X = X.fillna(0) # Handle NaN std cases

    y = diff_df["grok_step"]

    if len(y) > 2:
        model = sm.OLS(y, X).fit()
        with open(output_dir / "interaction_summary.txt", "w") as f:
            f.write(model.summary().as_text())
        return model
    else:
        with open(output_dir / "interaction_summary.txt", "w") as f:
            f.write("Not enough data to run OLS.")
        return None


def plot_grok_step(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        return

    diff_df = df[df["type"] == "difficulty"]
    if diff_df.empty:
        return

    plt.figure(figsize=(10, 6))
    for lvl in sorted(diff_df["collapse_level"].unique()):
        subset = diff_df[diff_df["collapse_level"] == lvl]
        # Aggregate by difficulty
        agg = subset.groupby("difficulty_score")["grok_step"].agg(["mean", "std"]).reset_index()
        plt.errorbar(agg["difficulty_score"], agg["mean"], yerr=agg["std"], label=f"Collapse {lvl}", marker='o')

    plt.xscale("log")
    plt.xlabel("Difficulty Score (prime^num_operands)")
    plt.ylabel("Grokking Step (Censored)")
    plt.title("Grokking Step vs Task Difficulty by Collapse Severity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "grok_step_vs_difficulty.png")
    plt.close()


def generate_markdown(df: pd.DataFrame, output_dir: Path):
    md_path = output_dir / "DIFFICULTY.md"

    # Read existing content to preserve static analysis
    content = ""
    if md_path.exists():
        with open(md_path, "r") as f:
            content = f.read()

    with open(md_path, "w") as f:
        f.write("# Difficulty vs Collapse Interaction\n\n")
        f.write("## Hypothesis\n")
        f.write("Does model collapse act as a fixed tax or a difficulty amplifier?\n\n")

        f.write("## Findings\n")
        f.write("See `interaction_summary.txt` for the full OLS regression output on the interaction term `difficulty * collapse_level`.\n")
        f.write("If the interaction coefficient is significantly positive, it implies collapse amplifies difficulty. Otherwise, it acts uniformly.\n\n")

        f.write("### Mechanistic Summary\n")
        f.write("Our initial local CPU smoke test demonstrates the functional setup of the `src/sweep/difficulty_sweep.py` pipeline crossing `prime` modulus and `num_operands` with `collapse_level`.\n")
        f.write("When executed at scale on HPC, the resulting `interaction_summary.txt` will reveal whether the collapse-induced delay in grokking acts independently (a fixed offset) or synergistically (an interaction amplifying the task difficulty tax).\n")
        f.write("Furthermore, the `Capacity Control` configuration holds difficulty fixed while varying `d_model` to help separate \"task too hard to learn on corrupted data\" from \"model too small to represent the true distribution under noise.\"\n\n")

        f.write("## Capacity Control\n")
        cap_df = df[df["type"] == "capacity"]
        if not cap_df.empty:
            agg = cap_df.groupby(["d_model", "collapse_level"])["grokked"].mean().unstack()
            f.write(agg.to_markdown())
            f.write("\n\n")

        f.write("![Grok Step vs Difficulty](grok_step_vs_difficulty.png)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/difficulty")
    parser.add_argument("--output-dir", type=str, default="analysis/difficulty")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(res_dir)
    if not df.empty:
        analyze_interaction(df, out_dir)
        plot_grok_step(df, out_dir)
        generate_markdown(df, out_dir)
        print(f"Analysis saved to {out_dir}")
    else:
        print("No results to analyze")

if __name__ == "__main__":
    main()
