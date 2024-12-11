import torch
import numpy as np


def post_process_peaks_binary(output, threshold=0.5, filter_peaks=False):
    output_probs = torch.sigmoid(output).squeeze()
    output_binary = (output_probs > threshold).float()
    all_peak_indices = torch.nonzero(output_binary).squeeze()

    if filter_peaks:
        # Ensure all_peak_indices is a 1D array or handle empty cases
        if all_peak_indices.numel() == 0:  # No peaks found
            peak_indices = np.array([])  # Empty array
            print("No peaks were found.")
        else:
            all_peak_indices = all_peak_indices.view(-1).cpu().numpy()  # Convert to numpy if non-empty
            # Remove extra peaks
            if len(all_peak_indices) > 1:  # more than one peak detected
                peak_differences = np.diff(all_peak_indices)  # Calculate differences between consecutive peaks
                if np.any(peak_differences < 40):  # Check if any peaks are closer than 40 samples
                    # Perform non-maximum suppression (only if necessary) by iterating through the peaks
                    filtered_peak_indices = []
                    i = 0
                    while i < len(all_peak_indices):
                        # Define the current window: 40 samples ahead from the current peak
                        window_start = all_peak_indices[i]
                        window_end = window_start + 40
                        # Find all peaks within the current window
                        window_peaks = all_peak_indices[
                            (all_peak_indices >= window_start) & (all_peak_indices < window_end)]
                        # Keep the peak with the highest probability in this window
                        if len(window_peaks) > 0:
                            max_peak = window_peaks[np.argmax(output_probs[window_peaks].detach().cpu().numpy())]
                            filtered_peak_indices.append(max_peak)
                        # Move the index to the next window (after the current window's end)
                        i += len(window_peaks)
                    # Convert filtered_peak_indices to numpy array for saving
                    peak_indices = np.array(filtered_peak_indices)
                else:
                    peak_indices = all_peak_indices
            else:  # only one peak detected
                peak_indices = all_peak_indices
    else:
        peak_indices = all_peak_indices

    return peak_indices
