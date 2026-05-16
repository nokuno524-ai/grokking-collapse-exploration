import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

class FourierFeatureAnalyzer:
    """Analyzes the frequency spectrum of representations during training."""

    @staticmethod
    def compute_frequency_spectrum(activations: Union[torch.Tensor, np.ndarray], dim: int = -1) -> np.ndarray:
        """
        Computes the Fourier power spectrum of activations.
        """
        if isinstance(activations, np.ndarray):
            x = torch.from_numpy(activations).float()
        else:
            x = activations.float()

        # Compute FFT along the specified dimension
        fft_result = torch.fft.rfft(x, dim=dim)

        # Calculate power spectrum (squared magnitude)
        power_spectrum = torch.abs(fft_result) ** 2

        # Average over all other dimensions
        reduce_dims = tuple(i for i in range(power_spectrum.dim()) if i != (dim % power_spectrum.dim()))
        if reduce_dims:
            mean_spectrum = power_spectrum.mean(dim=reduce_dims)
        else:
            mean_spectrum = power_spectrum

        # Normalize so sum = 1
        total_power = mean_spectrum.sum()
        if total_power > 0:
            mean_spectrum = mean_spectrum / total_power

        return mean_spectrum.cpu().numpy()

    @staticmethod
    def track_frequency_evolution(activation_history: List[Union[torch.Tensor, np.ndarray]], dim: int = -1) -> np.ndarray:
        """
        Tracks how the frequency spectrum changes over time.
        Returns a 2D array of shape (time_steps, frequencies).
        """
        spectra = []
        for x in activation_history:
            spectrum = FourierFeatureAnalyzer.compute_frequency_spectrum(x, dim=dim)
            spectra.append(spectrum)

        return np.stack(spectra)

    @staticmethod
    def detect_frequency_phase_shift(spectrum_history: np.ndarray, high_freq_threshold: int = 1, variance_threshold: float = 0.05) -> Optional[int]:
        """
        Detects the step where high-frequency components emerge.
        high_freq_threshold: indices >= this are considered "high frequency".
        variance_threshold: fraction of total power that must move to high frequencies to trigger detection.
        """
        if spectrum_history.shape[0] < 2:
            return None

        if high_freq_threshold >= spectrum_history.shape[1]:
            return None

        # Sum power in high frequencies for each step
        high_freq_power = np.sum(spectrum_history[:, high_freq_threshold:], axis=1)

        # Find where it first exceeds the threshold over the initial baseline
        baseline = np.mean(high_freq_power[:max(1, len(high_freq_power)//10)])

        for i in range(len(high_freq_power)):
            if high_freq_power[i] - baseline >= variance_threshold:
                # Check if it stays high
                if np.mean(high_freq_power[i:min(len(high_freq_power), i+5)]) - baseline >= variance_threshold * 0.8:
                    return i

        return None

    @staticmethod
    def plot_spectrum_heatmap(spectrum_history: np.ndarray, output_path: Union[str, Path], title: str = "Frequency Spectrum Evolution"):
        """
        Plots a heatmap of the frequency spectrum over time and saves it.
        """
        plt.figure(figsize=(10, 6))

        # spectrum_history is (steps, freqs)
        # Transpose for imshow (freqs on y-axis, steps on x-axis)
        plt.imshow(spectrum_history.T, aspect='auto', origin='lower', cmap='viridis',
                   interpolation='nearest')

        plt.colorbar(label='Normalized Power')
        plt.xlabel('Training Step (or checkpoint index)')
        plt.ylabel('Frequency Component')
        plt.title(title)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300)
        plt.close()
