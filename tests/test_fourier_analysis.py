import torch
import numpy as np
import pytest
import os
from src.fourier_analysis import FourierFeatureAnalyzer

def test_compute_frequency_spectrum():
    # Create a simple signal (sine wave)
    t = np.linspace(0, 1, 100)
    # Frequency 5
    signal = np.sin(2 * np.pi * 5 * t)

    analyzer = FourierFeatureAnalyzer()
    spectrum = analyzer.compute_frequency_spectrum(signal)

    assert spectrum.shape == (51,) # N//2 + 1
    # Peak should be at frequency 5
    peak_idx = np.argmax(spectrum)
    assert peak_idx == 5
    # Sum should be 1.0 (normalized)
    assert np.isclose(np.sum(spectrum), 1.0)

def test_track_frequency_evolution():
    # History of 10 signals
    history = [np.random.randn(100) for _ in range(10)]

    analyzer = FourierFeatureAnalyzer()
    spectra = analyzer.track_frequency_evolution(history)

    assert spectra.shape == (10, 51)

def test_detect_frequency_phase_shift():
    # Create a history where high frequencies suddenly emerge
    t = np.linspace(0, 1, 100)
    history = []

    # Steps 0-49: low freq only
    for _ in range(50):
        history.append(np.sin(2 * np.pi * 2 * t))

    # Steps 50-99: high freq added
    for _ in range(50):
        history.append(np.sin(2 * np.pi * 2 * t) + 2.0 * np.sin(2 * np.pi * 20 * t))

    analyzer = FourierFeatureAnalyzer()
    spectra = analyzer.track_frequency_evolution(history)

    shift_step = analyzer.detect_frequency_phase_shift(spectra, high_freq_threshold=10, variance_threshold=0.2)

    assert shift_step is not None
    assert 48 <= shift_step <= 52

def test_plot_spectrum_heatmap(tmp_path):
    spectra = np.random.rand(20, 30)
    output_path = tmp_path / "heatmap.png"

    analyzer = FourierFeatureAnalyzer()
    analyzer.plot_spectrum_heatmap(spectra, output_path)

    assert os.path.exists(output_path)
