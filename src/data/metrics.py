import numpy as np
import scipy.stats as stats
import collections
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

def compute_entropy(token_counts: Dict[int, int]) -> float:
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in token_counts.values()]
    return stats.entropy(probs)

def compute_distinct_n(sequence: List[int], n: int = 1) -> float:
    if len(sequence) < n:
        return 0.0
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    return len(set(ngrams)) / len(ngrams)

def compute_repetition_rate(sequence: List[int]) -> float:
    if len(sequence) < 2:
        return 0.0
    repeats = sum(1 for i in range(len(sequence)-1) if sequence[i] == sequence[i+1])
    return repeats / (len(sequence) - 1)

def compute_zipf_coefficient(token_counts: Dict[int, int]) -> float:
    if len(token_counts) < 2:
        return 0.0
    freqs = sorted(token_counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)
    # Fit log(freq) = -a * log(rank) + c
    log_freqs = np.log(freqs)
    log_ranks = np.log(ranks)
    slope, _, _, _, _ = stats.linregress(log_ranks, log_freqs)
    return -slope

def compute_jensen_shannon_divergence(p_counts: Dict[int, int], q_counts: Dict[int, int], vocab_size: int = 59) -> float:
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())

    if p_total == 0 or q_total == 0:
        return 1.0 # Max divergence

    p_probs = np.array([p_counts.get(i, 0) / p_total for i in range(vocab_size)])
    q_probs = np.array([q_counts.get(i, 0) / q_total for i in range(vocab_size)])

    m_probs = 0.5 * (p_probs + q_probs)

    jsd = 0.5 * stats.entropy(p_probs, m_probs) + 0.5 * stats.entropy(q_probs, m_probs)
    return float(jsd)

def evaluate_generation(targets: List[int], vocab_size: int = 59) -> Dict[str, float]:
    counts = collections.Counter(targets)
    metrics = {
        "entropy": compute_entropy(counts),
        "distinct_1": compute_distinct_n(targets, 1),
        "distinct_2": compute_distinct_n(targets, 2),
        "repetition_rate": compute_repetition_rate(targets),
        "zipf_coefficient": compute_zipf_coefficient(counts),
        "vocab_usage": len(counts) / vocab_size
    }
    return metrics

def generate_diversity_report(generations_data: List[Dict[str, Any]], out_dir: Path):
    """
    generations_data format:
    [
        {"gen_idx": 0, "targets": [1, 2, ...]},
        {"gen_idx": 1, "targets": [...]}
    ]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    vocab_size = 59 # Assuming default prime

    for i, gen in enumerate(generations_data):
        targets = gen["targets"]
        metrics = evaluate_generation(targets, vocab_size)
        metrics["gen_idx"] = gen["gen_idx"]

        # Calculate JS divergence from previous generation if available
        if i > 0:
            prev_targets = generations_data[i-1]["targets"]
            prev_counts = collections.Counter(prev_targets)
            curr_counts = collections.Counter(targets)
            metrics["jsd_from_prev"] = compute_jensen_shannon_divergence(curr_counts, prev_counts, vocab_size)
        else:
            metrics["jsd_from_prev"] = 0.0

        report.append(metrics)

    with open(out_dir / "diversity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plotting
    metrics_to_plot = ["entropy", "distinct_1", "distinct_2", "repetition_rate", "vocab_usage", "jsd_from_prev"]
    generations = [r["gen_idx"] for r in report]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        values = [r[metric] for r in report]
        axes[idx].plot(generations, values, marker='o')
        axes[idx].set_title(metric)
        axes[idx].set_xlabel("Generation")
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "diversity_metrics.png")
    plt.close()
