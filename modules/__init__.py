import torch
import os
import json
import glob
from BaseModel.model_architectures import GRUseq2seq
from config import RESULTS_PEAK_DETECTION


class Module:
    def __init__(self, module_name, architecture):
        """
        Initialize a new Module.

        :param module_name: Name of the module.
        :param architecture: A PyTorch nn.Module instance representing the architecture.
        :param checkpoints_directory: Directory where the weights from the original trained model are.
        :param weights_path: Path to load existing module weights.
        """
        self.module_name = module_name
        self.architecture = architecture  # This should be a PyTorch nn.Module or subclass
        self.module_weights_path = os.path.join(os.path.dirname(__file__), 'weights', f"{self.module_name}_weights.pth")
        self.training_info = {}
        # self.checkpoints_directory = checkpoints_directory

        if os.path.exists(self.module_weights_path):
            self.import_weights(self.module_weights_path)
        else:
            raise ValueError(f"Module {self.module_name} is not defined.")

    def import_original_weights(self, checkpoints_directory):
        """Load selected weights from an existing model checkpoint."""
        original_weights_path = glob.glob(os.path.join(checkpoints_directory, '*.ckpt'))[0]

        # Check if GPU is available and set map_location accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the full state dict (weights) from the file, mapping to GPU if available, otherwise to CPU
        full_state_dict = torch.load(original_weights_path, map_location=device)
        try:
            # Try to load all weights
            self.architecture.load_state_dict(full_state_dict, strict=True)
            print(f"Successfully loaded all weights from {original_weights_path} into {self.module_name}.")

        except RuntimeError as e:
            # If there’s a mismatch, load only the matching weights
            print(f"Full weight loading failed with error: {e}")
            print("Falling back to loading only matching weights...")

            # Get the current architecture's state dict
            current_state_dict = self.architecture.state_dict()

            # Filter out keys that match
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in current_state_dict}

            # Load the filtered weights into the current model
            self.architecture.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded partial weights from {original_weights_path} into {self.module_name}.")
        # # Get the current architecture's state dict
        # current_state_dict = self.architecture.state_dict()
        #
        # # Filter out keys that match
        # filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in current_state_dict}
        # # Load the filtered weights into the current model
        # self.architecture.load_state_dict(filtered_state_dict, strict=False)
        # print(f"Loaded partial weights from {original_weights_path} into {self.module_name}.")

    def load_training_info(self, checkpoints_directory):
        """
        Load training information for the module from the JSON file in original_checkpoints directory

        :param info_path: Optional path to the JSON file containing training information.
                          If not provided, it will look in the same directory as weights.
        """
        info_path = os.path.join(checkpoints_directory, 'trained_info.json')

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
            print(f"Training info loaded from {info_path} for module {self.module_name}.")
        else:
            print(f"No training info file found at {info_path}. Proceeding without training info.")

    def create_from_checkpoints(self, checkpoints_directory):
        """Imports original model's weights and saves module's weights and training info for future use."""

        self.import_original_weights(checkpoints_directory)
        self.load_training_info(checkpoints_directory)

        # os.makedirs(save_directory, exist_ok=True)
        save_directory = os.path.join(os.path.dirname(__file__), 'weights')
        weights_save_path = os.path.join(save_directory, f"{self.module_name}_weights.pth")
        # Save both the architecture's state_dict and any training info
        torch.save({
            'model_state_dict': self.architecture.state_dict(),
            'training_info': self.training_info
        }, weights_save_path)

        print(f"Module {self.module_name} saved with weights and training info at {weights_save_path}.")

    def import_weights(self, weights_path):
        """Load weights specific to this module."""
        # Check if GPU is available and set map_location accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the state dict, mapping to GPU if available, otherwise to CPU
        state_dict = torch.load(weights_path, map_location=device)
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
        # Define the architecture based on the trained model (GRUseq2seq)
        architecture = GRUseq2seq(
            n_features=1,  # Assuming 1 feature (ECG signal is univariate)
            hid_dim=64,
            n_layers=3,
            dropout=0.3,
            learning_rate=0.001,
            results_directory=None,
            gpu_id=None,
            bidirectional=True,
            task='classification',
            num_classes=1
        )

        # Initialize the Module and automatically load weights
        super().__init__(module_name="PeakDetectionGRU", architecture=architecture)

        # Set the device based on availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture.to(self.device)  # Move model to the appropriate device

    def peaks(self, signal, plot=False):
        """
        A method that applies the model to the input signal to detect peaks.

        :param signal: A tensor representing the input ECG signal for peak detection.
        :return: A numpy array with the indices of detected peaks.
        """
        # Prepare the signal for inference
        # Reshape the signal to match (batch_size=1, sequence_length, input_size=1)
        print(f"Original signal shape: {signal.shape}")
        if len(signal.shape) == 1:
            signal = signal.view(1, -1, 1)  # (1, sequence_length, 1)
        elif len(signal.shape) == 2:
            signal = signal.unsqueeze(0)  # Add batch dimension, becomes (1, sequence_length, 1)
        print(f"Reshaped signal shape: {signal.shape}")

        # Move the signal to the same device as the model
        signal = signal.to(self.device)
        lengths = [signal.size(1)]  # Length of the sequence

        # print(self.architecture.summary())
        self.architecture.to(self.device)
        self.architecture.eval()
        # Perform a forward pass (inference) with the model to get the output
        with torch.no_grad():
            output = self.forward(signal, lengths)

        # Post-process the output: Apply sigmoid and find the peak indices
        output_probs = torch.sigmoid(output).squeeze()
        print("Output probabilities:", output_probs.cpu().numpy())
        output_binary = (output_probs > 0.5).float()
        peak_indices = torch.nonzero(output_binary).squeeze().cpu().numpy()  # Get peak indices

        if plot:
            import matplotlib.pyplot as plt
            # print(signal.squeeze())
            plt.figure()
            plt.plot(signal.squeeze())
            plt.plot(list(peak_indices), signal.squeeze()[list(peak_indices)], 'o')
            plt.show()

        return peak_indices

    def create(self):
        """To run only once"""
        checkpoints_dir = os.path.join(RESULTS_PEAK_DETECTION, 'checkpoints',
                                       'GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26')
        PeakDetectionGRU().create_from_checkpoints(checkpoints_directory=checkpoints_dir)


# Example usage:
# create the module (once)
# PeakDetectionGRU().create()
import numpy as np
ecg_signal = np.load(r'C:\Users\Catia Bastos\dev\data\gib01_ecg\datasets\x\test\subject6_session10_segment2.npy')
signal = torch.tensor(ecg_signal).float()
module = PeakDetectionGRU()
detected_peaks = module.peaks(signal, plot=True)
print("Detected peak indices:", detected_peaks)
print(detected_peaks.shape)
# print(detected_peaks[:, 0])

# DEBUGGING CODE:
checkpoints_dir = os.path.join(RESULTS_PEAK_DETECTION, 'checkpoints',
                                       'GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26')
original_weights_path = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))[0]
checkpoint = torch.load(original_weights_path, map_location='cpu')

# Print out the keys in the checkpoint
print("Keys in the checkpoint:")
for key in checkpoint.keys():
    print(key)

# If the checkpoint includes the model's state_dict, print the layer names
if 'state_dict' in checkpoint:
    print("\nModel state_dict layers:")
    for layer_name in checkpoint['state_dict']:
        print(layer_name)

# If there's any additional metadata, print it to inspect for architecture details
if 'config' in checkpoint:  # Assuming config metadata was saved
    print("\nModel configuration:")
    print(checkpoint['config'])

for key, value in checkpoint.items():
    print(f"{key}: {type(value)}")
    if isinstance(value, dict):
        for sub_key in value:
            print(f"  {sub_key}: {type(value[sub_key])}")




