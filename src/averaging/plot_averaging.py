import json
import argparse
import os
import matplotlib.pyplot as plt

def plot_interpolation(results, output_dir):
    plt.figure(figsize=(10, 6))

    for condition, data in results.items():
        if "interpolation" not in data or not data["interpolation"]:
            continue

        alphas = [x["alpha"] for x in data["interpolation"]]
        accs = [x["acc"] for x in data["interpolation"]]

        plt.plot(alphas, accs, marker='o', label=condition)

    plt.title("Test Accuracy vs Interpolation Alpha")
    plt.xlabel("Alpha (0.0 = post-grok, 1.0 = pre-grok)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "accuracy_vs_alpha.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def plot_swa(results, output_dir):
    plt.figure(figsize=(10, 6))

    for condition, data in results.items():
        if "swa" not in data or not data["swa"]:
            continue

        windows = [x["window_size"] for x in data["swa"]]
        accs = [x["acc"] for x in data["swa"]]

        plt.plot(windows, accs, marker='o', label=condition)

    plt.title("SWA Test Accuracy vs Window Size")
    plt.xlabel("Window Size (number of checkpoints averaged)")
    plt.ylabel("Test Accuracy")
    plt.xticks(range(2, 6))
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "swa_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, default="analysis/averaging_results.json")
    parser.add_argument("--output-dir", type=str, default="analysis")
    args = parser.parse_args()

    with open(args.results_file, "r") as f:
        results = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_interpolation(results, args.output_dir)
    plot_swa(results, args.output_dir)

if __name__ == "__main__":
    main()
