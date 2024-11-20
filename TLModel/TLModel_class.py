import torch
import torch.nn as nn
import json
import os
from BaseModel import BaseModel
from modules import Module


class TLModel(BaseModel):
    def __init__(self, model_name, pretrained_modules=None, layers=None):
        """
        Initialize a TLModel using pretrained modules and potentially new layers.

        :param model_name: Name of the transfer-learning model.
        :param pretrained_modules: List of Module objects to be used in the TLModel.
        :param layers: List of nn.Modules to be added to the model for customization.
        """
        super(TLModel, self).__init__(model_name=model_name)

        # List to store pretrained modules
        self.modules_list = pretrained_modules if pretrained_modules else []

        # List to store additional layers
        self.layers = nn.ModuleList(layers) if layers else nn.ModuleList()

        # Combine pretrained modules and new layers
        self.full_model = nn.Sequential(*[mod.architecture for mod in self.modules_list], *self.layers)

        # To store training details if the model has been trained
        self.training_info = {}

    def add_pretrained_module(self, module):
        """
        Add a pretrained module to the TLModel
        :param module: instance of Module
        """
        if not isinstance(module, Module):
            raise ValueError("Only objects of type 'Module' can be added.")
        self.modules_list.append(module)
        self.full_model.add_module(module.module_name, module.architecture)

    def add_layers(self, layers):
        """
        Add additional layers (torch.nn.Modules) to the TLModel.
        :param layers: A list of nn.Module instances to add on top of the existing architecture.
        """
        if not all(isinstance(layer, nn.Module) for layer in layers):
            raise ValueError("All layers must be instances of nn.Module")
        for layer in layers:
            self.layers.append(layer)
            self.full_model.add_module(f"layer_{len(self.layers)}", layer)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the TLModel. Uses the full architectue (modules + layers).
        """
        return self.full_model(*args, **kwargs)

    def save_training_information(self, trainer, optimizer, train_dataset_name, val_loss, total_training_time,
                                  retraining):
        """Save information about the training process, including info about pretrained modules and the entire model."""
        self.training_info = {
            'train_dataset': train_dataset_name,
            'epochs': trainer.current_epoch,
            'optimizer': str(optimizer),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'validation_loss': val_loss,
            'training_time': total_training_time,  # If tracked manually
            'retraining': retraining,
            'pretrained_modules': [mod.model_name for mod in self.modules_list],
        }

        # Also save training info from the individual modules
        for mod in self.modules_list:
            self.training_info[f"{mod.model_name}_info"] = mod.training_info

    def save_model(self, save_directory):
        """
        Save the entire model (architecture, weights, training_info) to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        weights_save_path = os.path.join(save_directory, f"{self.model_name}_weights.pth")
        torch.save(self.full_model.state_dict(), weights_save_path)

        training_info_path = os.path.join(save_directory, f"{self.model_name}_training_info.json")
        with open(training_info_path, 'w') as f:
            json.dump(self.training_info, f, indent=4)

        print(f"TLModel {self.model_name} saved with weights at {weights_save_path}.")

    def freeze_modules(self):
        """
        Freeze all pretrained modules, allowing only new layers to be trained.
        """

        for mod in self.modules_list:
            mod.freeze()
        print(f"All pretrained modules in {self.module_name} have been frozen.")

    def unfreeze_modules(self):
        """
        Unfreeze all pretrained modules to allow them to be retrained.
        """
        for mod in self.modules_list:
            mod.unfreeze()
        print(f"All pretrained modules in {self.module_name} have been unfrozen.")

