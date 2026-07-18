import os
import json

def generate_html_report(stats_path="analysis/statistics_results.json", plots_dir="analysis/plots", output_path="analysis/report.html"):
    """
    Generate an HTML report summarizing the findings of the attention analysis.
    """
    stats_data = {}
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grokking and Model Collapse: Attention Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .plot-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); width: calc(50% - 35px); }}
            .plot-card.full {{ width: 100%; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #e9ecef; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Attention Dynamics: Grokking vs Model Collapse</h1>
        <p>This report investigates how attention patterns differ between models that grok (generalize successfully) and those that experience model collapse (degenerative training on synthetic data).</p>

        <h2>1. Attention Entropy Evolution</h2>
        <p>Attention entropy measures how diffuse (high entropy) or focused (low entropy) attention heads become during training. Grokking often involves the formation of sharp, sparse circuits (low entropy).</p>
        <div class="plot-card full">
            <img src="plots/entropy_evolution.png" alt="Attention Entropy Evolution">
        </div>

        <h2>2. Circuit Formation</h2>
        <p>Basic attention circuits involve attending to the same token (self-attention) or different tokens (cross-attention). Here we observe how these patterns emerge under different conditions.</p>
        <div class="container">
            <div class="plot-card">
                <h3>Pure (Grokking)</h3>
                <img src="plots/circuit_pure.png" alt="Circuit Formation - Pure">
            </div>
            <div class="plot-card">
                <h3>High Collapse</h3>
                <img src="plots/circuit_high_collapse.png" alt="Circuit Formation - High Collapse">
            </div>
        </div>

        <h2>3. Final Attention State Comparison</h2>
        <p>A comparison of the attention state at the end of training across all severity conditions.</p>
        <div class="container">
            <div class="plot-card">
                <img src="plots/entropy_comparison.png" alt="Entropy Comparison">
            </div>
            <div class="plot-card">
                <img src="plots/circuit_comparison.png" alt="Circuit Comparison">
            </div>
        </div>

        <h2>4. Attention Entropy Surface</h2>
        <div class="plot-card full">
            <img src="plots/entropy_3d.png" alt="3D Entropy Surface">
        </div>

        <h2>5. Statistical Analysis</h2>
        <h3>Correlation: Attention Entropy vs Test Accuracy</h3>
        <pre>{json.dumps(stats_data.get('correlations', {}), indent=2)}</pre>

        <h3>Significance: Pure vs Collapsed (Final Entropy)</h3>
        <pre>{json.dumps(stats_data.get('significance', {}), indent=2)}</pre>

        <h3>Regression: Predicting Grokking from Early Entropy</h3>
        <pre>{json.dumps(stats_data.get('regression', {}), indent=2)}</pre>

    </body>
    </html>
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Report generated successfully at {output_path}")

if __name__ == "__main__":
    generate_html_report()
