import torch
import os
import BaseModel as bm
from config import RESULTS_PEAK_DETECTION
from Module_class import Module


class PeakDetectionGRU(Module):
    def __init__(self):

        # Initialize the Module and automatically load weights
        super().__init__(module_name="PeakDetectionGRU", architecture_class=bm.GRUseq2seq)

        # Set the device based on availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def peaks(self, sig, plot=False, threshold=0.5, min_distance=40):
        """
        A method that applies the model to the input signal to detect peaks.
        :param sig: A tensor representing the input ECG signal for peak detection (tensor of shape [seq_len]).
        :param plot: bool
        :param threshold: threshold for the probability above which a signal sample should be classified as "peak"
        :param min_distance: minimum distance (in samples) between 2 consecutive peaks.
        :return: A numpy array with the indices of detected peaks.
        """
        # Prepare the signal for inference
        # Reshape the signal to match (batch_size=1, sequence_length, input_size=1)
        print(f"Original signal shape: {sig.shape}")
        signal = sig.clone().detach()
        signal = signal.view(1, -1, 1)  # Add batch dimension: from torch.Size([7560]) to torch.Size([1, 7560, 1])
        signal = signal.to(self.device)  # Move the signal to the same device as the model
        lengths = [signal.size(1)]  # Length of the sequence

        # print(self.architecture.summary())
        self.architecture.to(self.device)  # Move model to the appropriate device
        self.architecture.eval()
        # Perform a forward pass (inference) with the model to get the output
        with torch.no_grad():
            output = self.forward(signal, lengths)
            output_probs = torch.sigmoid(output).squeeze()
            output_binary = (output_probs > threshold).float()
            all_peak_indices = torch.nonzero(output_binary).squeeze().cpu().numpy()

            # Ensure all_peak_indices is a 1D array or handle empty cases
            if all_peak_indices.size == 0:  # No peaks found
                filtered_peak_indices = np.array([])  # Empty array
                print("No peaks were detected.")
            else:
                print(f"{all_peak_indices.size} peaks were detected. Proceeding to remove extra peaks, if existing.")
                # Remove extra peaks
                if len(all_peak_indices) > 1:
                    peak_differences = np.diff(all_peak_indices)  # Calculate differences between consecutive peaks
                    if np.any(peak_differences < min_distance):
                        # Perform non-maximum suppression (only if necessary) by iterating through the peaks
                        filtered_peak_indices = []
                        i = 0
                        while i < len(all_peak_indices):
                            # Define the current window: min_distance samples ahead from the current peak
                            window_start = all_peak_indices[i]
                            window_end = window_start + min_distance
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
                        filtered_peak_indices = np.array(filtered_peak_indices)
                        # Find
                    else:
                        filtered_peak_indices = all_peak_indices
                else:
                    filtered_peak_indices = all_peak_indices

        print(
            f"{filtered_peak_indices.size} peaks found. Peak indices for the provided signal: {filtered_peak_indices}")

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(signal.squeeze())
            plt.plot(list(filtered_peak_indices), signal.squeeze()[list(filtered_peak_indices)], 'o')
            plt.show()

        return filtered_peak_indices


# Example usage:
# create the module (once)
checkpoints_dir = os.path.join(RESULTS_PEAK_DETECTION, 'checkpoints',
                               'GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26')
PeakDetectionGRU().create_from_checkpoints(checkpoints_directory=checkpoints_dir,
                                           whole_model=True,
                                           architecture_class=bm.GRUseq2seq)
import numpy as np

ecg_signal = np.load(r'C:\Users\Catia Bastos\dev\data\gib01_ecg\datasets\x\test\subject6_session10_segment2.npy')
signal = torch.tensor(ecg_signal).float()
module = PeakDetectionGRU()
detected_peaks = module.peaks(signal, plot=True)
print("Detected peak indices:", detected_peaks)
print(detected_peaks.shape)
