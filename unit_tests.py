import torch
import numpy as np
from production_models import ECGPeakDetector, post_process_peaks_binary


def test_peak_detection():
    model = ECGPeakDetector()
    signal = torch.rand(1, 100, 1)  # Random input signal
    peaks = model.peak_detect(signal)
    assert isinstance(peaks, np.ndarray)  # Ensure output is an array
    assert peaks.ndim == 1  # Ensure it's a 1D array of indices
