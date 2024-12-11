import torch
import os
import json
import glob
import architectures.biosignals_architectures as bm
from config import RESULTS_PEAK_DETECTION


class Module:
    def __init__(self, module_name, architecture_class, architecture=None):
        """
        Initialize a new Module.
        :param module_name: Name of the module.
        """
        self.module_name = module_name
        self.architecture = architecture  # This should be a PyTorch nn.Module or subclass
        self.module_weights_path = os.path.join(os.path.dirname(__file__), '../modules/weights',
                                                f"{self.module_name}_weights.pth")
        self.training_info = {}
        self.architecture_class = architecture_class

        if os.path.exists(self.module_weights_path):
            self.import_weights(self.module_weights_path)  # here self.architecture is defined, so no worries
        else:
            print(f"Module {self.module_name} is not defined.")

    def create_from_checkpoints(self, checkpoints_directory, whole_model=True, architecture_class=None,
                                architecture=None):
        """
        Imports original model's weights and saves module's weights and training info for future use.

        :param checkpoints_directory: Directory containing checkpoint files.
        :param whole_model: Boolean indicating if the whole model (architecture + weights) should be loaded.
        :param architecture_class: Optional class inheriting from BaseModel to specify the architecture type
                                   when `whole_model=True`.
        :param architecture: If `whole_model=False`, this should be an instantiated PyTorch nn.Module or subclass.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        original_ckpt_file = glob.glob(os.path.join(checkpoints_directory, '*.ckpt'))[0]

        if whole_model:
            if architecture_class is None:
                raise ValueError("architecture_class must be provided when whole_model=True.")

            # Load the full model to extract weights and architecture parameters
            full_model = architecture_class.load_from_checkpoint(original_ckpt_file)
            self.architecture = full_model
            architecture_params = {
                'n_features': full_model.n_features,
                'hid_dim': full_model.hid_dim,
                'n_layers': full_model.n_layers,
                'bidirectional': full_model.bidirectional,
                'dropout': full_model.dropout,
                'learning_rate': full_model.learning_rate,
                'results_directory': full_model.results_directory
            }
        else:
            if architecture is None:
                raise ValueError("architecture must be provided since whole_model=False.")
            else:
                self.architecture = architecture
                # Load the full state dict (weights) from the file, mapping to GPU if available, otherwise to CPU
                full_state_dict = torch.load(original_ckpt_file, map_location=device)
                filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in self.architecture.state_dict()}
                self.architecture.load_state_dict(filtered_state_dict, strict=False)
                architecture_params = {
                    'n_features': self.architecture.n_features,
                    'hid_dim': self.architecture.hid_dim,
                    'n_layers': self.architecture.n_layers,
                    'bidirectional': self.architecture.bidirectional,
                    'dropout': self.architecture.dropout,
                    'learning_rate': self.architecture.learning_rate,
                    'results_directory': self.architecture.results_directory
                }

        # Load training information for the module from the JSON file in original_checkpoints directory
        info_path = os.path.join(checkpoints_directory, 'trained_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
            print(f"Training info loaded from {info_path} for module {self.module_name}.")
        else:
            print(f"No training info file found at {info_path}. Proceeding without training info.")

        # Save the weights and architecture parameters together in a single file
        save_directory = os.path.join(os.path.dirname(__file__), '../modules/weights')
        os.makedirs(save_directory, exist_ok=True)
        weights_save_path = os.path.join(save_directory, f"{self.module_name}_weights.pth")
        torch.save({
            'model_state_dict': self.architecture.state_dict(),
            'architecture_params': architecture_params,
            'training_info': self.training_info
        }, weights_save_path)

        print(f"Module {self.module_name} saved with weights and training info at {weights_save_path}.")

    def import_weights(self, weights_path):
        """Load weights specific to this module."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(weights_path, map_location=device)
        if self.architecture is None:
            architecture_params = state_dict.get('architecture_params')
            if architecture_params is None:
                raise ValueError("No architecture parameters found in the checkpoint file.")
            # Initialize the architecture with the loaded parameters
            self.architecture = self.architecture_class(**architecture_params)
        self.architecture.load_state_dict(state_dict['model_state_dict'], strict=True)
        self.training_info = state_dict.get('training_info', {})
        print(f"Loaded module weights from {weights_path} into {self.module_name}.")

    def forward(self, *args, **kwargs):
        """
        Forward pass through the module. This assumes that the architecture has been defined.
        It delegates the forward pass to the architecture's own forward method, since it is assumed to be a PyTorch
        nn.Module instance.

        :param args: Input arguments for the architecture's forward pass.
        :param kwargs: Keyword arguments for the architecture's forward pass.
        """
        if self.architecture is None:
            raise ValueError("Architecture must be defined before calling forward.")

        return self.architecture(*args, **kwargs)

    def freeze(self):
        """Freeze the weights of the module."""
        if self.architecture is None:
            raise ValueError("Architecture must be defined before freezing.")

        for param in self.architecture.parameters():
            param.requires_grad = False
        print(f"All layers of module {self.module_name} are now frozen.")

    def unfreeze(self):
        """Unfreeze the weights of the module."""
        if self.architecture is None:
            raise ValueError("Architecture must be defined before unfreezing.")

        for param in self.architecture.parameters():
            param.requires_grad = True
        print(f"All layers of module {self.module_name} are now unfrozen.")

    def summary(self):
        """Provide a summary of the module's architecture and parameters."""
        if self.architecture is None:
            return f"Module {self.module_name} has no defined architecture."

        total_params = sum(p.numel() for p in self.architecture.parameters())
        return f"Module {self.module_name} structure:\n{str(self.architecture)}\nTotal parameters: {total_params}"


# Now we create a Module for peak detection based on your pre-trained GRUseq2seq model
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
                    if np.any(
                            peak_differences < min_distance):  # Check if any peaks are closer than min_distance samples
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
