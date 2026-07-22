import os
import sys
import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.attention_evolution import run_attention_analysis
from visualization.weight_analysis import run_all_weight_analysis
from analysis.circuit_detection import track_circuit_formation, analyze_circuit_correlations

def create_pdf_report(output_dir: str, pdf_path: str):
    """Aggregate generated images and CSV data into a multi-page PDF report."""
    print(f"Generating PDF report at {pdf_path}...")

    with PdfPages(pdf_path) as pdf:
        # Title Page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Grokking & Model Collapse:\nMechanistic Interpretability Report',
                 ha='center', va='center', fontsize=24, weight='bold')
        fig.text(0.5, 0.5, 'Generated Analysis Suite',
                 ha='center', va='center', fontsize=16)

        # Add simple stats if available
        csv_path = os.path.join(output_dir, "circuit_correlation_summary.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            stats_text = "Circuit Formation Summary:\n\n"
            for _, row in df.iterrows():
                stats_text += f"{row['condition']}: Grok={row['grokking_step']}, Rank Drop={row['rank_drop_step']}\n"
            fig.text(0.5, 0.3, stats_text, ha='center', va='top', fontsize=12, family='monospace')

        pdf.savefig(fig)
        plt.close()

        # Collect generated images
        img_paths = []

        # Weight norm trajectories
        norm_img = os.path.join(output_dir, "weight_norm_trajectories.png")
        if os.path.exists(norm_img):
            img_paths.append(("Weight Norm Evolutions", norm_img))

        # SVD Spectra
        svd_imgs = sorted(glob.glob(os.path.join(output_dir, "svd_spectrum_*.png")))
        for img in svd_imgs:
            cond = os.path.basename(img).split('_')[2]
            img_paths.append((f"Singular Value Spectrum ({cond})", img))

        # Distributions
        dist_imgs = sorted(glob.glob(os.path.join(output_dir, "weight_dist_*.png")))
        for img in dist_imgs:
            cond = os.path.basename(img).split('_')[2]
            layer = os.path.basename(img).split('_')[3].split('.')[0]
            img_paths.append((f"Weight Distribution: {layer} ({cond})", img))

        # Attention Maps
        attn_imgs = sorted(glob.glob(os.path.join(output_dir, "*_attention_step_*.png")))
        for img in attn_imgs:
            cond = os.path.basename(img).split('_')[0]
            step = os.path.basename(img).split('.')[0].split('_')[-1]
            img_paths.append((f"Attention Map: {cond} Step {step}", img))

        # Add images to PDF
        for title, img_path in img_paths:
            try:
                img = Image.open(img_path)
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title, fontsize=16)
                pdf.savefig(fig)
                plt.close()
            except Exception as e:
                print(f"Error adding {img_path} to PDF: {e}")

def main():
    output_dir = "results/analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    print("Running Attention Analysis...")
    try:
        run_attention_analysis()
    except Exception as e:
        print(f"Error in attention analysis: {e}")

    print("Running Weight Analysis...")
    try:
        run_all_weight_analysis()
    except Exception as e:
        print(f"Error in weight analysis: {e}")

    print("Running Circuit Detection...")
    try:
        df = track_circuit_formation()
        analyze_circuit_correlations(df, output_dir=output_dir)
    except Exception as e:
        print(f"Error in circuit detection: {e}")

    pdf_path = "results/comprehensive_report.pdf"
    create_pdf_report(output_dir, pdf_path)
    print(f"Successfully completed analysis suite. Report saved to {pdf_path}.")

if __name__ == "__main__":
    main()
