import torch
from torch import nn
import os
import json
from BaseModel.model_architectures import GRUseq2seq


class Module:
    def __init__(self, module_name, architecture, weights_path):
        """
        Initialize a new Module.

        :param module_name: Name of the module.
        :param architecture: A PyTorch nn.Module instance representing the architecture.
        :param weights_path: Path to the pretrained weights file. If provided, weights will be loaded.
        """
        self.module_name = module_name
        self.architecture = architecture  # This should be a PyTorch nn.Module or subclass
        self.weights_path = weights_path
        self.training_info = {}
        # Load weights immediately if the path is provided
        if weights_path:
            self.import_weights(weights_path)

    def import_weights(self, weights_path):
        """
        Import weights from a model (possibly a more complex model), using only those that match this architecture.
        TODO: if there are multiple options of weights that fit my architecture, how do i choose?
        """
        # Load the full state dict (weights) from the file
        full_state_dict = torch.load(weights_path)

        # Get the current architecture's state dict
        current_state_dict = self.architecture.state_dict()

        # Filter out keys that match
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in current_state_dict}

        # Load the filtered weights into the current model
        self.architecture.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded partial weights from {weights_path} into {self.module_name}.")

        # Optionally load training information if it's available in the state dict
        if 'training_info' in full_state_dict:
            self.training_info = full_state_dict['training_info']
            print(f"Loaded training info from {weights_path}.")

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

    def save(self, save_directory):
        """
        Save the module's architecture and weights to the specified directory.

        :param save_directory: Directory where the architecture and weights should be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        weights_save_path = os.path.join(save_directory, f"{self.module_name}_weights.pth")
        # Save both the architecture's state_dict and any training info
        torch.save({
            'model_state_dict': self.architecture.state_dict(),
            'training_info': self.training_info
        }, weights_save_path)

        print(f"Module {self.module_name} saved with weights and training info at {weights_save_path}.")

    def load_training_info(self, info_path=None):
        """
        Load training information for the module from a JSON file.

        :param info_path: Optional path to the JSON file containing training information.
                          If not provided, it will look in the same directory as weights.
        """
        if not info_path:
            # Infer the path from the weights file if not provided
            weights_dir = os.path.dirname(self.weights_path)
            info_path = os.path.join(weights_dir, 'trained_info.json')

            # Check if the inferred or provided path exists
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
            print(f"Training info loaded from {info_path} for module {self.module_name}.")
        else:
            print(f"No training info file found at {info_path}. Proceeding without training info.")

    def freeze(self):
        """
        Freeze the weights of the module, preventing updates during training.
        """
        if self.architecture is None:
            raise ValueError("Architecture must be defined before freezing.")

        for param in self.architecture.parameters():
            param.requires_grad = False
        print(f"All layers of module {self.module_name} are now frozen.")

    def unfreeze(self):
        """
        Unfreeze the weights of the module, allowing updates during training.
        """
        if self.architecture is None:
            raise ValueError("Architecture must be defined before unfreezing.")

        for param in self.architecture.parameters():
            param.requires_grad = True
        print(f"All layers of module {self.module_name} are now unfrozen.")

    def summary(self):
        """
        Provides a more detailed summary of the architecture, useful for debugging.
        """
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
            batch_size=1,  # For inference, we use a batch size of 1
            results_directory=None,
            gpu_id=None,
            bidirectional=True,
            task='classification',
            num_classes=1
        )

        # Instead of passing the weights every time, we hardcode the path here.
        weights_path = os.path.join(os.path.dirname(__file__), 'weights/peak_detection_gru_weights.pth')

        # Initialize the Module and automatically load weights
        super().__init__(module_name="PeakDetectionGRU", architecture=architecture, weights_path=weights_path)

    def detect_peaks(self, signal):
        """
        A method that applies the model to the input signal to detect peaks.

        :param signal: A tensor representing the input ECG signal for peak detection.
        :return: A numpy array with the indices of detected peaks.
        """
        # Prepare the signal for inference
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)  # Add batch dimension if necessary
        lengths = [signal.size(1)]  # Length of the sequence

        # Perform a forward pass (inference) with the model to get the output
        with torch.no_grad():
            output = self.forward(signal, lengths)

        # Post-process the output: Apply sigmoid and find the peak indices
        output_probs = torch.sigmoid(output).squeeze()
        output_binary = (output_probs > 0.5).float()
        peak_indices = torch.nonzero(output_binary).squeeze().cpu().numpy()  # Get peak indices

        return peak_indices

# Example usage:
# Assuming you have an input signal as a PyTorch tensor:
# signal = torch.tensor(your_ecg_signal).float()
# peak_module = PeakDetectionGRU()
# detected_peaks = peak_module.detect_peaks(signal)
# print("Detected peak indices:", detected_peaks)

