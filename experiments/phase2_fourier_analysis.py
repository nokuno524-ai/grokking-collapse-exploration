import argparse
import json
import os
import torch
import numpy as np

from src.data import generate_modular_arithmetic, DatasetConfig
from src.model import ModularArithmeticTransformer
from src.train import train, TrainConfig

def analyze_fourier(model: ModularArithmeticTransformer, top_k: int = 5):
    """
    Measure how concentrated the Fourier spectrum is on the top-k frequencies.
    Uses squared magnitude (energy) instead of absolute magnitude.
    """
    W = model.token_embed.weight.detach()  # (prime, d_model)
    spectrum = torch.fft.fft(W, dim=0).abs() ** 2  # Energy
    avg_spectrum = spectrum.mean(dim=1)  # Average energy across embedding dimensions

    # Exclude DC component
    avg_spectrum = avg_spectrum[1:]

    total_energy = avg_spectrum.sum()
    if total_energy < 1e-10:
        return 0.0, avg_spectrum.cpu().numpy()

    top_energy = avg_spectrum.topk(min(top_k, len(avg_spectrum))).values.sum()
    concentration = (top_energy / total_energy).item()

    return concentration, avg_spectrum.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Fourier Concentration Analysis Phase 2")
    parser.add_argument("--collapse_level", type=float, default=0.15)
    parser.add_argument("--noise_rate", type=float, default=0.15)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/phase2_fourier_analysis")
    parser.add_argument("--max_steps", type=int, default=10000, help="For testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    for condition in ["pure", "collapse", "noise"]:
        results[condition] = []
        for seed in range(42, 42 + args.num_seeds):
            print(f"--- Fourier Analysis: {condition.capitalize()} Seed {seed} ---")

            c_level = args.collapse_level if condition == "collapse" else 0.0
            n_rate = args.noise_rate if condition == "noise" else 0.0

            train_config = TrainConfig(
                prime=59,
                train_fraction=0.3,
                collapse_level=c_level,
                collapse_severity=0.5 if condition == "collapse" else 0.0,
                noise_fraction=n_rate,
                seed=seed,
                output_dir=os.path.join(args.output_dir, f"{condition}_s{seed}"),
                condition_name=condition,
                max_steps=args.max_steps,
                save_every=100,  # Ensure granular checkpointing for temporal dynamics
            )

            # Run the training
            state = train(train_config)

            # Recompute Fourier exactly according to the Phase 2 requirements (using energy instead of absolute value)
            # using the saved checkpoints
            fourier_history = []
            heatmap_data = []

            # Evaluate checkpoints if they were saved, otherwise use the history from TrainState
            for entry in state.history:
                step = entry['step']
                ckpt_path = os.path.join(train_config.output_dir, condition, f"checkpoint_{step}.pt")
                if os.path.exists(ckpt_path):
                    model = ModularArithmeticTransformer(prime=59)
                    # Use try-except as noted in the memory
                    try:
                        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    except Exception:
                        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                    model.load_state_dict(checkpoint['model_state'])

                    concentration, spectrum_energy = analyze_fourier(model)
                    fourier_history.append({"step": step, "concentration": concentration})
                    heatmap_data.append(spectrum_energy.tolist())

            # Ensure the directory exists
            os.makedirs(os.path.join(args.output_dir, condition), exist_ok=True)

            # Save heatmap data to disk
            heatmap_file = os.path.join(args.output_dir, condition, f"heatmap_s{seed}.json")
            with open(heatmap_file, "w") as f:
                json.dump({"steps": [h['step'] for h in fourier_history], "spectrum_energy": heatmap_data}, f)

            results[condition].append({
                "seed": seed,
                "fourier_history": fourier_history
            })

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Fourier Analysis completed successfully.")

if __name__ == "__main__":
    main()
