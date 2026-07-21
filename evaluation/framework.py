import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import SyntheticDataGenerator
from metrics.collapse_predictor import extract_metrics_from_data, CollapsePredictor

def generate_report(results: Dict[str, Any], output_dir: str):
    """
    Generate JSON and Markdown reports from the evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save Markdown
    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("# Synthetic Data Quality & Collapse Prediction Report\n\n")

        f.write("## Overview\n")
        f.write(f"- Dataset generated with seed: {results.get('seed', 'N/A')}\n")
        f.write(f"- Number of collapse levels evaluated: {len(results.get('levels', {}))}\n\n")

        f.write("## Detailed Metrics by Collapse Level\n\n")
        for level, data in results.get("levels", {}).items():
            f.write(f"### Level: {level}\n")
            metrics = data.get("metrics", {})
            for k, v in metrics.items():
                f.write(f"- **{k}**: {v:.4f}\n")
            f.write("\n")

        if "prediction" in results:
            f.write("## Collapse Prediction Results\n\n")
            f.write("### Model Training\n")
            f.write(f"- **Cross-validation accuracy**: {results['prediction'].get('cv_accuracy', 0.0):.4f}\n\n")

            f.write("### Feature Importance\n")
            for k, v in results['prediction'].get('feature_importance', {}).items():
                f.write(f"- **{k}**: {v:.4f}\n")
            f.write("\n")

            f.write("### Collapse Thresholds\n")
            for k, v in results['prediction'].get('thresholds', {}).items():
                f.write(f"- **{k}**: {v:.4f}\n")
            f.write("\n")

def run_evaluation(seed: int = 42, prime: int = 59, output_dir: str = "evaluation_results"):
    """
    Run evaluation across multiple collapse levels and train a predictor.
    """
    print(f"Running evaluation (seed={seed}, prime={prime})...")
    generator = SyntheticDataGenerator(prime=prime, seed=seed)

    collapse_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = {
        "seed": seed,
        "prime": prime,
        "levels": {}
    }

    metrics_list = []
    labels = []

    for level in collapse_levels:
        print(f"Generating data for collapse level {level}...")

        # We consider level > 0.3 as a dataset that would cause collapse (label=1)
        label = 1 if level > 0.3 else 0
        labels.append(label)

        # We'll use nucleus sampling as our artifact generator for this test
        orig, synth = generator.generate(
            collapse_level=level,
            collapse_severity=0.8,
            sampling_strategy="nucleus",
            sampling_kwargs={"p": 0.9}
        )

        # If there's no synthetic data, just compare orig to a copy of itself to get baseline metrics
        if not synth:
            synth_to_eval = list(orig)
        else:
            synth_to_eval = synth

        print(f"Extracting metrics for collapse level {level}...")
        metrics = extract_metrics_from_data(orig, synth_to_eval)

        results["levels"][str(level)] = {
            "label": label,
            "metrics": metrics
        }
        metrics_list.append(metrics)

    # Train predictor if we have enough points (though 5 is very small, we'll try for demo)
    print("Training collapse predictor...")
    predictor = CollapsePredictor(model_type="rf")
    cv_acc = predictor.train(metrics_list, labels)

    importances = predictor.get_feature_importance()

    # Get thresholds for the top 3 features
    top_features = list(importances.keys())[:3]
    thresholds = {}
    for feature in top_features:
        thresholds[feature] = predictor.find_collapse_threshold(feature, metrics_list, labels)

    results["prediction"] = {
        "cv_accuracy": cv_acc,
        "feature_importance": importances,
        "thresholds": thresholds
    }

    print(f"Writing report to {output_dir}...")
    generate_report(results, output_dir)
    print("Evaluation complete.")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic data quality and predict model collapse.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prime", type=int, default=59, help="Prime modulus for arithmetic data")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save the report")

    args = parser.parse_args()
    run_evaluation(seed=args.seed, prime=args.prime, output_dir=args.output_dir)
