import pytest
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_quality import (
    ngram_diversity,
    token_frequency_analysis,
    perplexity_estimation,
    repetition_detection,
    collapse_score
)

def test_ngram_diversity():
    # Sequence with all unique tokens (high diversity)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    div = ngram_diversity(tokens, max_n=2)
    assert div[1] == 1.0 # 8 unique unigrams / 8 total
    assert div[2] == 1.0 # 7 unique bigrams / 7 total

    # Highly repetitive sequence (low diversity)
    tokens_rep = [1, 2, 1, 2, 1, 2, 1, 2]
    div_rep = ngram_diversity(tokens_rep, max_n=2)
    assert div_rep[1] == 2/8 # 2 unique (1, 2) / 8 total
    assert div_rep[2] == 2/7 # unique bigrams: (1,2), (2,1) / 7 total

def test_token_frequency_analysis():
    # Uniform distribution
    tokens_uniform = [1, 2, 3, 4, 1, 2, 3, 4]
    ref_dist = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}

    metrics = token_frequency_analysis(tokens_uniform, reference_distribution=ref_dist)
    # KL should be very close to 0
    assert abs(metrics["kl_divergence"]) < 1e-5

    # Skewed distribution (simulating collapse)
    tokens_skewed = [1, 1, 1, 1, 1, 2, 3, 4]
    metrics_skewed = token_frequency_analysis(tokens_skewed, reference_distribution=ref_dist)
    # KL should be positive
    assert metrics_skewed["kl_divergence"] > 0

    # Zipf should be positive for skewed
    assert metrics_skewed["zipf_coefficient"] > 0

def test_perplexity_estimation():
    # Very predictable
    tokens_predictable = [1, 2, 1, 2, 1, 2, 1, 2]
    ppl_pred = perplexity_estimation(tokens_predictable)

    # Random
    tokens_random = [1, 3, 2, 4, 2, 1, 4, 3]
    ppl_rand = perplexity_estimation(tokens_random)

    # Predictable sequence should have lower perplexity
    assert ppl_pred < ppl_rand

def test_repetition_detection():
    # No repetitions of length 3
    tokens_clean = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rep_clean = repetition_detection(tokens_clean, sequence_length=3)
    assert rep_clean == 0.0

    # One repeated sequence of length 3: (1,2,3)
    tokens_rep = [1, 2, 3, 4, 5, 1, 2, 3, 9]
    rep_dirty = repetition_detection(tokens_rep, sequence_length=3)
    assert rep_dirty > 0.0

def test_collapse_score():
    tokens_clean = [1, 2, 3, 4, 5, 6, 7, 8]
    tokens_collapsed = [1, 1, 1, 1, 2, 1, 1, 1]

    ref = {i: 1/8 for i in range(1, 9)}

    score_clean = collapse_score(tokens_clean, reference_distribution=ref)
    score_collapsed = collapse_score(tokens_collapsed, reference_distribution=ref)

    # Collapsed should have a higher collapse score
    assert score_collapsed > score_clean
