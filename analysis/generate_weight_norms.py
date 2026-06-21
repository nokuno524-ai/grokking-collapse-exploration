import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_weight_norms(results_dir="results"):
    """
    Finds all results.json files in the results directory,
    parses the step and weight_norm metrics, and returns a dictionary.
    Format: { condition_name: {"steps": [], "norms": []} }
    """
    data = {}
    pattern = os.path.join(results_dir, "*/results.json")
    files = glob.glob(pattern)

    for fpath in files:
        condition = os.path.basename(os.path.dirname(fpath))
        with open(fpath, "r") as f:
            res = json.load(f)

        if "history" in res:
            steps = [entry["step"] for entry in res["history"]]
            norms = [entry["weight_norm"] for entry in res["history"]]
            data[condition] = {"steps": steps, "norms": norms}

    return data

def plot_weight_norms(data, save_path):
    """
    Plots the weight norm trajectories across conditions.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for condition, metrics in data.items():
        plt.plot(metrics["steps"], metrics["norms"], label=condition, linewidth=2)

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("L2 Weight Norm", fontsize=12)
    plt.title("Weight Norm Trajectories across Collapse Levels", fontsize=14)
    plt.legend(title="Condition")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # If there are no results yet, let's create some dummy data for testing
    import numpy as np
    dummy_data = {
        "pure": {"steps": list(range(100, 2000, 100)), "norms": np.linspace(20, 100, 19).tolist()},
        "medium_collapse": {"steps": list(range(100, 2000, 100)), "norms": np.linspace(20, 50, 19).tolist()}
    }

    # Try parsing first
    parsed_data = parse_weight_norms()
    if not parsed_data:
        print("No results.json found, using dummy data for verification.")
        parsed_data = dummy_data

    plot_weight_norms(parsed_data, "visualizations/weight_norms.png")
    print("Saved weight norms plot to visualizations/weight_norms.png")
