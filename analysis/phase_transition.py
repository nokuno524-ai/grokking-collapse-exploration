import json
import numpy as np
import os
import glob

def find_grokking_step(history):
    """
    Detect the grokking point (phase transition).
    Uses the jump in test accuracy.
    """
    steps = [h["step"] for h in history]
    test_accs = [h["test_acc"] for h in history]

    # Calculate discrete derivative of test acc
    d_acc = np.diff(test_accs)

    # If the model never achieves good accuracy, it didn't grok
    if max(test_accs) < 0.9:
        return -1, None

    # Find the largest jump in accuracy
    max_jump_idx = np.argmax(d_acc)
    grokking_step = steps[max_jump_idx + 1]

    # Calculate confidence interval based on step frequency
    step_size = steps[1] - steps[0] if len(steps) > 1 else 100
    ci_lower = max(0, grokking_step - step_size)
    ci_upper = grokking_step + step_size

    return grokking_step, (ci_lower, ci_upper)

def analyze_all_transitions():
    results = {}
    for result_file in glob.glob("results/*/results.json"):
        condition = os.path.basename(os.path.dirname(result_file))
        with open(result_file, "r") as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            continue

        grok_step, ci = find_grokking_step(history)

        # Also check weight norm plateau (norm at grokking step)
        norm_plateau = -1
        if grok_step != -1:
            for h in history:
                if h["step"] >= grok_step:
                    norm_plateau = h["weight_norm"]
                    break

        results[condition] = {
            "grokking_step": grok_step,
            "confidence_interval": ci,
            "weight_norm_at_transition": norm_plateau
        }

    print(json.dumps(results, indent=2))

    # Save structured results
    with open("results/phase_transitions.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    analyze_all_transitions()
