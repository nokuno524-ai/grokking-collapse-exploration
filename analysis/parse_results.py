import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_results():
    data = []

    # Read the phase transition results to get accurate grokking steps
    phase_transitions = {}
    pt_path = "results/phase_transitions.json"
    if os.path.exists(pt_path):
        with open(pt_path, "r") as f:
            phase_transitions = json.load(f)

    for result_file in glob.glob("results/*/results.json"):
        condition = os.path.basename(os.path.dirname(result_file))
        # Skip pure grid or multi_seed as they have different structure sometimes
        if condition in ["grid", "multi_seed", "seed_sweep", "noise_baseline", "scarcity_baseline", "exp_c_grid"]:
            continue

        with open(result_file, "r") as f:
            res = json.load(f)

        final_test_acc = res.get("final_test_acc", 0.0)
        final_train_acc = res.get("final_train_acc", 0.0)
        final_weight_norm = res.get("final_weight_norm", 0.0)

        # Override with phase transition grokking step if available, else from results.json
        grok_step = res.get("grokking_step", -1)
        if condition in phase_transitions:
            grok_step = phase_transitions[condition]["grokking_step"]

        collapse_level = res.get("config", {}).get("collapse_level", 0.0)

        data.append({
            "Condition": condition,
            "Collapse Level": collapse_level,
            "Final Train Acc": final_train_acc,
            "Final Test Acc": final_test_acc,
            "Final Weight Norm": final_weight_norm,
            "Grokking Step": grok_step if grok_step != -1 else "N/A",
            "Grokked": grok_step != -1
        })

    df = pd.DataFrame(data)
    df = df.sort_values("Collapse Level")

    print("Summary Table:")
    print(df.to_markdown(index=False))

    # Save CSV
    df.to_csv("results/parsed_summary.csv", index=False)

    # Multi-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Final Accuracies
    df_melt = pd.melt(df, id_vars=['Condition'], value_vars=['Final Train Acc', 'Final Test Acc'])
    sns.barplot(data=df_melt, x='Condition', y='value', hue='variable', ax=axes[0])
    axes[0].set_title("Final Accuracy by Condition")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis='x', rotation=45)

    # Panel 2: Weight Norm
    sns.barplot(data=df, x='Condition', y='Final Weight Norm', ax=axes[1], color="skyblue")
    axes[1].set_title("Final Weight Norm by Condition")
    axes[1].tick_params(axis='x', rotation=45)

    # Panel 3: Grokking Step
    df_grokked = df[df["Grokked"] == True]
    if not df_grokked.empty:
        sns.barplot(data=df_grokked, x='Condition', y='Grokking Step', ax=axes[2], color="salmon")
    axes[2].set_title("Grokking Step (if achieved)")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("results/summary_multi_panel.png", dpi=300)
    print("Saved summary table and multi-panel figure.")

if __name__ == "__main__":
    parse_results()
