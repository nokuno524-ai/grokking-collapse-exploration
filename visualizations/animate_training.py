import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, Any, List

# Setting backend to non-interactive
matplotlib.use('Agg')

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_results(results_dir: str = "results") -> Dict[str, Dict[str, Any]]:
    base_path = Path(results_dir)
    data = {}
    if not base_path.exists():
        print(f"Warning: {results_dir} does not exist.")
        return data

    for p in base_path.iterdir():
        if p.is_dir():
            json_file = p / "results.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data[p.name] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: could not parse {json_file}")
    return data

def extract_metric(history: List[Dict[str, Any]], metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        return np.array([]), np.array([])

    steps = []
    values = []
    for entry in history:
        if "step" in entry and metric_name in entry:
            steps.append(entry["step"])
            values.append(entry[metric_name])

    return np.array(steps), np.array(values)

def create_animation(condition: str = "pure", data_dir: str = "results", output_path: str = "visualizations/training_animation.mp4"):
    """Creates an animation of training metrics for a specific condition."""
    data = load_results(data_dir)

    if condition not in data:
        print(f"Warning: Condition '{condition}' not found. Cannot animate.")
        return

    history = data[condition].get("history", [])
    if not history:
        print(f"Warning: No history found for '{condition}'.")
        return

    steps, loss_vals = extract_metric(history, "test_loss")
    _, acc_vals = extract_metric(history, "test_acc")
    _, norm_vals = extract_metric(history, "weight_norm")

    if len(steps) == 0:
        print("No steps to animate.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_loss = axes[0]
    ax_acc = axes[1]
    ax_norm = axes[2]

    # Setup axes
    ax_loss.set_title("Test Loss")
    ax_loss.set_xlabel("Steps")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, max(steps) * 1.05)
    ax_loss.set_ylim(min(loss_vals) * 0.9, max(loss_vals) * 1.1)

    ax_acc.set_title("Test Accuracy")
    ax_acc.set_xlabel("Steps")
    ax_acc.set_xlim(0, max(steps) * 1.05)
    ax_acc.set_ylim(-0.05, 1.05)

    ax_norm.set_title("Weight Norm")
    ax_norm.set_xlabel("Steps")
    ax_norm.set_xlim(0, max(steps) * 1.05)
    ax_norm.set_ylim(0, max(norm_vals) * 1.1)

    # Initialize lines
    line_loss, = ax_loss.plot([], [], lw=2, color="#e74c3c")
    line_acc, = ax_acc.plot([], [], lw=2, color="#2ecc71")
    line_norm, = ax_norm.plot([], [], lw=2, color="#3498db")

    time_text = fig.suptitle("", fontsize=16)

    def init():
        line_loss.set_data([], [])
        line_acc.set_data([], [])
        line_norm.set_data([], [])
        time_text.set_text("")
        return line_loss, line_acc, line_norm, time_text

    def update(frame):
        # frame is the index of the step
        current_steps = steps[:frame+1]

        line_loss.set_data(current_steps, loss_vals[:frame+1])
        line_acc.set_data(current_steps, acc_vals[:frame+1])
        line_norm.set_data(current_steps, norm_vals[:frame+1])

        time_text.set_text(f"Condition: {condition} | Step: {steps[frame]}")

        return line_loss, line_acc, line_norm, time_text

    # Skip some frames if history is very long to speed up animation
    num_frames = len(steps)
    step_size = max(1, num_frames // 100) # Keep around 100 frames
    frames_to_render = list(range(0, num_frames, step_size))
    if frames_to_render[-1] != num_frames - 1:
        frames_to_render.append(num_frames - 1)

    ani = animation.FuncAnimation(
        fig, update, frames=frames_to_render,
        init_func=init, blit=True, interval=50
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Try MP4 first, fallback to GIF
    try:
        if output_path.endswith('.mp4'):
            ani.save(output_path, writer='ffmpeg', fps=20, dpi=150)
            print(f"Saved animation to {output_path}")
        else:
            ani.save(output_path, writer='pillow', fps=20, dpi=150)
            print(f"Saved animation to {output_path}")
    except Exception as e:
        print(f"Failed to save {output_path} with primary writer. Falling back to GIF via pillow. Error: {e}")
        fallback_path = output_path.rsplit('.', 1)[0] + '.gif'
        ani.save(fallback_path, writer='pillow', fps=20, dpi=150)
        print(f"Saved animation to {fallback_path}")

    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="pure")
    parser.add_argument("--output", type=str, default="visualizations/training_animation.gif")
    args = parser.parse_args()

    create_animation(args.condition, output_path=args.output)
