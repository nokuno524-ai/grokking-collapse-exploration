import os
from pathlib import Path

def generate_dashboard(output_dir: str = "analysis"):
    """Generate an HTML dashboard combining all visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grokking vs Collapse Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            h1, h2 { color: #333; }
            .container { max-width: 1200px; margin: auto; }
            .section { background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; background: #fff; }
            .description { font-size: 16px; color: #555; line-height: 1.5; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Model Collapse vs Grokking</h1>
            <p class="description">Dashboard visualizing the findings of the investigation into how dataset collapse impacts the ability of transformers to grok modular arithmetic.</p>

            <div class="section">
                <h2>1. Capability Emergence</h2>
                <p class="description">Test accuracy across training steps. Notice how severe collapse completely prevents the grokking phase transition from occurring.</p>
                <img src="figures/capability_curves.png" alt="Capability Emergence Curves">
            </div>

            <div class="section">
                <h2>2. Phase Transition Timing</h2>
                <p class="description">Comparison of the exact step when the model successfully groks the dataset (reaches >95% test accuracy).</p>
                <img src="figures/phase_transitions.png" alt="Phase Transitions">
            </div>

            <div class="section">
                <h2>3. Weight Norm Trajectories</h2>
                <p class="description">Weight norm reduction correlates with collapse severity. Pure models show a characteristic growth followed by stabilization during grokking.</p>
                <img src="figures/weight_norms.png" alt="Weight Norms">
            </div>

            <div class="section">
                <h2>4. Attention Pattern Evolution</h2>
                <p class="description">How information routing changes. Pure grokked models develop sharp, structured attention, while collapsed models remain diffuse.</p>
                <img src="figures/attention_patterns.png" alt="Attention Patterns">
            </div>

            <div class="section">
                <h2>5. Loss Landscape Geometry</h2>
                <p class="description">The loss landscape shifts from wide, flat minima (conducive to generalization/grokking) to sharp, rough minima under severe collapse.</p>
                <img src="figures/loss_landscape.png" alt="Loss Landscape">
            </div>
        </div>
    </body>
    </html>
    """

    output_path = Path(output_dir) / "dashboard.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Dashboard generated at: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
