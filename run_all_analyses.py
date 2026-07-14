import os
import argparse
from src.model import ModularArithmeticTransformer
from analysis.fourier_circuits import analyze_fourier_circuits, plot_fourier_heatmaps, compare_runs
from analysis.attention_evolution import analyze_attention_evolution
from analysis.j_space_probe import compare_j_space

def main():
    parser = argparse.ArgumentParser(description="Run all mechanistic analyses.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing run results")
    parser.add_argument("--prime", type=int, default=59, help="Prime modulus")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Directory to save analysis plots and reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Identify run directories
    # A run dir should contain checkpoint files
    run_dirs = []
    for root, dirs, files in os.walk(args.results_dir):
        if any(f.startswith("checkpoint_") and f.endswith(".pt") for f in files):
            run_dirs.append(root)

    print(f"Found {len(run_dirs)} runs for analysis.")

    if len(run_dirs) == 0:
        print("No runs found. Exiting.")
        return

    # 1. Fourier circuit evolution for all runs
    # To avoid overwhelming memory, we'll pick the first two for deep time-series analysis
    sample_runs = run_dirs[:2] if len(run_dirs) > 0 else []

    for run_dir in sample_runs:
        run_name = os.path.basename(run_dir)
        print(f"Processing Fourier evolution for {run_name}...")
        ckpts = sorted([os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("checkpoint_") and f.endswith(".pt")],
                       key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

        model = ModularArithmeticTransformer(prime=args.prime)
        history = []
        for ckpt in ckpts:
            res = analyze_fourier_circuits(ckpt, model, args.prime)
            history.append(res)

        out_path = os.path.join(args.output_dir, run_name, "fourier")
        plot_fourier_heatmaps(history, out_path)

    # 2. Attention Pattern Evolution
    print("Processing Attention Pattern Evolution...")
    analyze_attention_evolution(run_dirs[:3], ModularArithmeticTransformer, args.prime, os.path.join(args.output_dir, "attention"))

    # 3. Compare specific runs if they exist (e.g., pure vs severe_collapse)
    pure_dir = os.path.join(args.results_dir, "pure")
    collapsed_dir = os.path.join(args.results_dir, "severe_collapse")

    if os.path.exists(pure_dir) and os.path.exists(collapsed_dir):
        print("Comparing Grokked (pure) vs Collapsed (severe_collapse)...")
        compare_runs(pure_dir, collapsed_dir, ModularArithmeticTransformer, args.prime, os.path.join(args.output_dir, "comparison"))
        compare_j_space(pure_dir, collapsed_dir, ModularArithmeticTransformer, args.prime, os.path.join(args.output_dir, "comparison"))
    else:
        print("Could not find 'pure' and 'severe_collapse' directories for direct comparison.")

    print("All analyses completed successfully.")

if __name__ == "__main__":
    main()
