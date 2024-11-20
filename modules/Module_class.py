import torch
import os
import json
import glob


class Module:
    def __init__(self, module_name, architecture_class, architecture=None):
        """
        Initialize a new Module.
        :param module_name: Name of the module.
        """
        self.module_name = module_name
        self.architecture = architecture  # This should be a PyTorch nn.Module or subclass
        self.module_weights_path = os.path.join(os.path.dirname(__file__), 'weights',
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
        save_directory = os.path.join(os.path.dirname(__file__), 'weights')
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


