"""
Progress measures for grokking analysis.
Based on Chan et al. (2023) "Progress Measures for Grokking via Mechanistic Interpretability"
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

SEVERITY_ORDER = [
    "pure",
    "low_collapse",
    "medium_collapse",
    "high_collapse",
    "severe_collapse",
]


def iter_conditions_by_severity(results_dir: Path) -> Iterable[Path]:
    """Yield condition subdirectories of results_dir in severity order, then any extras alphabetically."""
    by_name = {p.name: p for p in results_dir.iterdir() if p.is_dir()}
    for name in SEVERITY_ORDER:
        if name in by_name:
            yield by_name.pop(name)
    for name in sorted(by_name):
        yield by_name[name]


def load_results(condition_dir: Path) -> Dict:
    """Load results JSON for a condition."""
    with open(condition_dir / "results.json") as f:
        return json.load(f)


def compute_generalization_gap(history: List[Dict]) -> List[float]:
    """
    Compute the train/test loss gap at every logged step.

    Note: this is the generalization gap, NOT the "excluded loss" progress measure
    from Chan et al. (2023). True excluded loss requires projecting out specific
    Fourier components from the model's logits and recomputing cross-entropy, which
    needs the model and dataset — not just the history record. Until that is wired
    up end-to-end, the generalization gap is the honest summary we can produce here.
    """
    return [entry["test_loss"] - entry["train_loss"] for entry in history]


def detect_phase_transition(
    history: List[Dict], metric: str = "test_acc", threshold: float = 0.9
) -> Optional[int]:
    """
    Detect the step at which a phase transition occurs.
    Returns the step number or None if no transition detected.
    """
    for entry in history:
        if entry.get(metric, 0) >= threshold:
            return entry["step"]
    return None


def compute_learning_speed(
    history: List[Dict], metric: str = "test_acc", window: int = 10
) -> List[Dict]:
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


def memorization_phase_fourier_slope(history: List[Dict]) -> Optional[float]:
    """Leading-indicator candidate.

    The audit found that `fourier_concentration` is a *lagging* indicator at
    our resolution: it crosses a threshold concurrent with test_acc, not
    before. But the *slope* of fourier_concentration during the memorization
    phase (after train_acc ≥ 0.99 and before test_acc ≥ 0.95) might be
    predictive — high slope means circuit formation is already underway while
    test_acc is still flat.

    Returns the per-1000-step slope of fourier_concentration over the
    memorization-only window, or None if that window has fewer than 3 points.

    A pre-registered claim to test on held-out runs: runs that grok have
    significantly larger memorization-phase Fourier slopes than runs that
    fail to grok.
    """
    if not history:
        return None
    mem_idx = None
    grok_idx = None
    for i, e in enumerate(history):
        if mem_idx is None and e.get("train_acc", 0) >= 0.99:
            mem_idx = i
        if e.get("test_acc", 0) >= 0.95:
            grok_idx = i
            break
    if mem_idx is None:
        return None
    end_idx = grok_idx if grok_idx is not None else len(history) - 1
    window = history[mem_idx : end_idx + 1]
    if len(window) < 3:
        return None
    steps = np.array([e["step"] for e in window], dtype=float)
    fc = np.array([e.get("fourier_concentration", 0.0) for e in window], dtype=float)
    # OLS slope per 1000 steps
    A = np.column_stack([np.ones_like(steps), steps])
    sol, *_ = np.linalg.lstsq(A, fc, rcond=None)
    slope_per_step = float(sol[1])
    return slope_per_step * 1000.0


def weight_norm_decay_slope(history: List[Dict]) -> Optional[float]:
    """Leading-indicator candidate #2.

    Weight norm peaks during memorization and falls during cleanup. The slope
    of ``log(weight_norm)`` post-memorization-completion has been argued
    (Liu 2022, Chan 2023) to be a more reliable signal than Fourier
    concentration. Returns slope per 1000 steps in log-space, or None if the
    post-mem window is too short.
    """
    if not history:
        return None
    mem_idx = None
    for i, e in enumerate(history):
        if e.get("train_acc", 0) >= 0.99:
            mem_idx = i
            break
    if mem_idx is None:
        return None
    window = history[mem_idx:]
    if len(window) < 5:
        return None
    steps = np.array([e["step"] for e in window], dtype=float)
    wn = np.array([max(e.get("weight_norm", 1e-3), 1e-3) for e in window], dtype=float)
    A = np.column_stack([np.ones_like(steps), steps])
    sol, *_ = np.linalg.lstsq(A, np.log(wn), rcond=None)
    return float(sol[1]) * 1000.0


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

    # Find circuit formation onset: sustained growth of Fourier concentration.
    # Track running mean over the last 5 eval steps; trigger when it crosses 0.1
    # while still monotonically increasing across the window. The previous
    # "50% jump in a single step" heuristic almost never fired because Fourier
    # concentration grows smoothly during circuit formation rather than
    # spiking.
    circuit_onset = None
    window = 5
    for i in range(window, len(history) + 1):
        recent = [
            history[j].get("fourier_concentration", 0) for j in range(i - window, i)
        ]
        running_mean = sum(recent) / window
        monotonic = all(recent[k] <= recent[k + 1] for k in range(window - 1))
        if running_mean > 0.1 and monotonic:
            circuit_onset = history[i - 1]["step"]
            break

    # Compute key metrics
    max_weight_norm = max(e.get("weight_norm", 0) for e in history) if history else 0
    min_weight_norm = (
        min(e.get("weight_norm", float("inf")) for e in history) if history else 0
    )

    return {
        "phases_detected": True,
        "memorization_complete_step": mem_complete_step,
        "circuit_formation_onset": circuit_onset,
        "grokking_step": grok_step,
        "delay_mem_to_grok": (
            (grok_step - mem_complete_step)
            if (grok_step and mem_complete_step)
            else None
        ),
        "max_weight_norm": max_weight_norm,
        "min_weight_norm": min_weight_norm,
        "weight_norm_reduction": max_weight_norm - min_weight_norm,
        "memorization_phase_fourier_slope": memorization_phase_fourier_slope(history),
        "weight_norm_decay_slope": weight_norm_decay_slope(history),
    }


def generate_comparison_table(results_dir: Path) -> str:
    """Generate a markdown comparison table of all conditions."""
    rows = []
    rows.append(
        "| Condition | Grokked? | Grokking Step | Final Test Acc | Fourier Conc. | Embedding Rank |"
    )
    rows.append(
        "|-----------|----------|---------------|----------------|---------------|----------------|"
    )

    for condition_dir in iter_conditions_by_severity(results_dir):
        try:
            results = load_results(condition_dir)
            name = condition_dir.name
            grokked = "✅" if results.get("grokked") else "❌"
            step = results.get("grokking_step", "N/A")
            acc = f"{results.get('final_test_acc', 0):.4f}"
            fc = f"{results.get('final_fourier_concentration', 0):.3f}"
            rank = f"{results.get('final_embedding_rank', 0):.1f}"
            rows.append(f"| {name} | {grokked} | {step} | {acc} | {fc} | {rank} |")
        except Exception:
            rows.append(f"| {condition_dir.name} | Error | - | - | - | - |")

    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    print("Grokking-Collapse Experiment Analysis")
    print("=" * 60)

    for condition_dir in iter_conditions_by_severity(results_dir):
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
