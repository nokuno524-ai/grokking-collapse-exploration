import json
from pathlib import Path
from src.paper_figures.fig1_grok_curves import generate_grok_curves
from src.paper_figures.fig2_weight_norm import generate_weight_norm_curves
from src.paper_figures.fig3_cliff import generate_cliff_figure
from src.paper_figures.fig4_gap import generate_gap_figure
from src.paper_figures.fig5_combined import generate_combined_figure

def generate_all():
    registry_path = Path("results/registry.json")
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not registry_path.exists():
        print(f"Registry not found at {registry_path}")
        return

    generate_grok_curves(registry_path, output_dir)
    generate_weight_norm_curves(registry_path, output_dir)
    generate_cliff_figure(registry_path, output_dir)
    generate_gap_figure(registry_path, output_dir)
    generate_combined_figure(registry_path, output_dir)
    print("All figures generated successfully.")

if __name__ == "__main__":
    generate_all()
