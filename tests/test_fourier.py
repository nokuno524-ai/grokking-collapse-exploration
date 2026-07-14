import torch
import pytest
from analysis.fourier_circuits import get_2d_fourier_transform

def test_2d_fft():
    # create a 2x2 grid with known pattern
    # e.g., an alternating grid
    grid = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).view(1, 1, 1, 2, 2)
    fft_mag = get_2d_fourier_transform(grid)

    # DC component should be 0
    assert torch.isclose(fft_mag[0, 0, 0, 0, 0], torch.tensor(0.0), atol=1e-5)

    # Nyquist component should be large (4.0)
    assert torch.isclose(fft_mag[0, 0, 0, 1, 1], torch.tensor(4.0), atol=1e-5)
