"""
Progress measures for grokking analysis.
Based on Chan et al. (2023) "Progress Measures for Grokking via Mechanistic Interpretability"
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_results(condition_dir: Path) -> Dict:
    """Load results JSON for a condition."""
    with open(condition_dir / "results.json") as f:
        return json.load(f)


def compute_excluded_loss(history: List[Dict], prime: int = 59) -> List[float]:
    """
    Compute excluded loss — the loss attributable to specific Fourier components.
    This is a progress measure from Chan et al.
    
    High excluded loss = model relies on those components = circuit formation in progress.
    """
    excluded = []
    for entry in history:
        test_loss = entry["test_loss"]
        train_loss = entry["train_loss"]
        # Simplified proxy: gap between train and test loss
        excluded.append(test_loss - train_loss)
    return excluded


def detect_phase_transition(history: List[Dict], metric: str = "test_acc",
                            threshold: float = 0.9) -> Optional[int]:
    """
    Detect the step at which a phase transition occurs.
    Returns the step number or None if no transition detected.
    """
    for entry in history:
        if entry.get(metric, 0) >= threshold:
            return entry["step"]
    return None


def compute_learning_speed(history: List[Dict], metric: str = "test_acc",
                           window: int = 10) -> List[Dict]:
    """Compute rate of change of a metric over a sliding window."""
    speeds = []
    for i in range(len(history)):
        if i < window:
            speed = 0.0
        else:
            current = history[i].get(metric, 0)
            past = history[i - window].get(metric, 0)
            steps_diff = history[i]["step"] - history[i - window]["step"]
            speed = (current - past) / max(steps_diff, 1) * 1000  # per 1000 steps
        speeds.append({"step": history[i]["step"], f"{metric}_speed": speed})
    return speeds


def classify_run(history: List[Dict]) -> str:
    """
    Classify the training run into:
    - grokking: test_acc > 0.95 at the end
    - memorization: train_acc > 0.95 but test_acc < 0.95 at the end
    - collapse: mode_collapse > 0.5 or kl_div > 1.0 (indicating distribution shifted significantly)
    - normal/failed: otherwise
    """
    if not history:
        return "failed"

    final = history[-1]

    # If grokking
    if final.get("test_acc", 0) > 0.95:
        return "grokking"

    # If mode collapse or severe distribution shift detected
    if final.get("mode_collapse", 0) > 0.5 or final.get("kl_div", 0) > 1.0:
        return "collapse"

    # If memorizing only
    if final.get("train_acc", 0) > 0.95 and final.get("test_acc", 0) < 0.95:
        return "memorization"

    return "normal"


def analyze_grokking_trajectory(history: List[Dict]) -> Dict:
    """
    Analyze the full grokking trajectory, identifying phases.
    
    Phase 1: Memorization (train_acc rises, test_acc stays low)
    Phase 2: Circuit formation (fourier_concentration rises)
    Phase 3: Cleanup/grokking (test_acc jumps, weight_norm decreases)
    """
    if not history:
        return {"phases_detected": False}
    
    # Find memorization completion (train_acc > 0.99)
    mem_complete_step = None
    for entry in history:
        if entry.get("train_acc", 0) > 0.99:
            mem_complete_step = entry["step"]
            break
    
    # Find grokking step
    grok_step = detect_phase_transition(history, "test_acc", 0.95)
    
    # Find circuit formation onset (fourier_concentration starts rising)
    circuit_onset = None
    for i in range(1, len(history)):
        prev_fc = history[i-1].get("fourier_concentration", 0)
        curr_fc = history[i].get("fourier_concentration", 0)
        if curr_fc > 0.1 and curr_fc > prev_fc * 1.5:
            circuit_onset = history[i]["step"]
            break

    classification = classify_run(history)
    
    # Compute key metrics
    max_weight_norm = max(e.get("weight_norm", 0) for e in history) if history else 0
    min_weight_norm = min(e.get("weight_norm", float('inf')) for e in history) if history else 0
    
    return {
        "phases_detected": True,
        "classification": classification,
        "memorization_complete_step": mem_complete_step,
        "circuit_formation_onset": circuit_onset,
        "grokking_step": grok_step,
        "delay_mem_to_grok": (grok_step - mem_complete_step) if (grok_step and mem_complete_step) else None,
        "max_weight_norm": max_weight_norm,
        "min_weight_norm": min_weight_norm,
        "weight_norm_reduction": max_weight_norm - min_weight_norm,
    }


def generate_comparison_table(results_dir: Path) -> str:
    """Generate a markdown comparison table of all conditions."""
    rows = []
    rows.append("| Condition | Class | Grokked? | Grokking Step | Final Test Acc | Fourier Conc. | Mode Collapse |")
    rows.append("|-----------|-------|----------|---------------|----------------|---------------|---------------|")
    
    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        try:
            results = load_results(condition_dir)
            history = results.get("history", [])
            name = condition_dir.name
            run_class = classify_run(history)
            grokked = "✅" if results.get("grokked") else "❌"
            step = results.get("grokking_step", "N/A")
            acc = f"{results.get('final_test_acc', 0):.4f}"
            fc = f"{results.get('final_fourier_concentration', 0):.3f}"
            mode_collapse = history[-1].get('mode_collapse', 0.0) if history else 0.0
            mc = f"{mode_collapse:.3f}"
            rows.append(f"| {name} | {run_class} | {grokked} | {step} | {acc} | {fc} | {mc} |")
        except Exception as e:
            rows.append(f"| {condition_dir.name} | Error | - | - | - | - | - |")
    
    return "\n".join(rows)


if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    
    print("Grokking-Collapse Experiment Analysis")
    print("=" * 60)
    
    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        try:
            results = load_results(condition_dir)
            analysis = analyze_grokking_trajectory(results.get("history", []))
            print(f"\n{condition_dir.name}:")
            for k, v in analysis.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"\n{condition_dir.name}: Error - {e}")
    
    print("\n" + "=" * 60)
    print(generate_comparison_table(results_dir))
