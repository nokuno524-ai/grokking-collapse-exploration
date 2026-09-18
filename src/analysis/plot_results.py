import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_dilution(df, out_path):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="collapse_level", y="grokking_step", marker="o")
    plt.title("Grokking Cliff vs Synthetic Mixture Ratio")
    plt.xlabel("Fraction of Collapsed Data")
    plt.ylabel("Grokking Step")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def plot_scaling(df, out_path):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="train_fraction", y="final_test_acc", hue="condition", marker="o")
    plt.title("Final Test Accuracy vs Dataset Size")
    plt.xlabel("Train Fraction")
    plt.ylabel("Final Test Accuracy")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dilution-csv", type=str)
    parser.add_argument("--scaling-csv", type=str)
    parser.add_argument("--out-dir", type=str, default="analysis/")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)

    if args.dilution_csv:
        df = pd.read_csv(args.dilution_csv)
        plot_dilution(df, f"{args.out_dir}/dilution_cliff.png")

    if args.scaling_csv:
        df = pd.read_csv(args.scaling_csv)
        plot_scaling(df, f"{args.out_dir}/scaling_acc.png")
