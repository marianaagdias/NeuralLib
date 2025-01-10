from NeuralLib.architectures import Architecture
import torch
import json
import yaml
import os
import torch.nn as nn


class TLModel(Architecture):
    """
    Transfer Learning Model that supports layer-wise weight injection and flexible training strategies.
    """
    def __init__(self, architecture_class, **kwargs):
        """
        Initialize the architecture just like any other model.
        :param architecture_class: The architecture class (e.g., GRUseq2seq).
        :param kwargs: Architecture hyperparameters.
        """
        super().__init__(model_name=kwargs.get('model_name', 'TransferLearningModel'))
        self.architecture_class = architecture_class
        self.hyperparams = kwargs
        self.model = self.architecture_class(**self.hyperparams)
        # Dynamically set kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        print(f"✅ TransferLearningModel initialized with architecture: {architecture_class.__name__}")

    def inject_weights(self, layer_mapping):
        """
        Inject weights into specific layers of the model.
        :param layer_mapping: Dict mapping target layers in TLModel to source state_dicts.
                              Example: {'model.gru_layers.0': source_GRU_layer.state_dict()}
        """
        print(f"layer mapping items: {layer_mapping.items()}")
        for layer_name, source_state_dict in layer_mapping.items():
            print(f"layer_name: {layer_name}")
            try:
                # Navigate to the target layer
                if '.' in layer_name:
                    print(f"hi - {layer_name}")
                    parts = layer_name.split('.')
                    current_attr = self.model
                    for part in parts:
                        if part.isdigit():
                            current_attr = current_attr[int(part)]
                        else:
                            current_attr = getattr(current_attr, part)

                    # Load the state_dict into the target layer
                    current_attr.load_state_dict(source_state_dict)
                    print(f"✅ Weights injected into {layer_name} from source layer.")
                else:
                    raise ValueError(f"Layer '{layer_name}' not found in TLModel.")
            except (AttributeError, IndexError) as e:
                raise ValueError(f"Failed to inject weights into {layer_name}: {e}")

    def forward(self, X, lengths=None):
        """
        Forward pass through the architecture.
        :param X: Input tensor.
        :param lengths: Sequence lengths for variable-length input support.
        :return: Model output.
        """
        return self.model(X, lengths)

    def train_tl(self, *args, **kwargs):
        """
        Train the model using injected pre-trained weights.
        This method uses 'train_from_scratch' for TLModel.
        """
        print("⚙️ Training Transfer Learning Model with pre-trained weights...")
        super().train_from_scratch(*args, **kwargs)

    # Delegate training_step to self.model
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    # Delegate validation_step to self.model
    def validation_step(self, batch, batch_idx):
        # Pass `self.log` to the model so it logs in the Trainer's context
        self.model.log = self.log
        return self.model.validation_step(batch, batch_idx)

    # Delegate configure_optimizers to self.model
    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def freeze_layers(self, layers_to_freeze):
        """
        Freeze specific layers.
        :param layers_to_freeze: List of layer names to freeze.
        """
        for layer_name in layers_to_freeze:
            if hasattr(self.model, layer_name):
                for param in getattr(self.model, layer_name).parameters():
                    param.requires_grad = False
                print(f"✅ Layer '{layer_name}' has been frozen.")
            else:
                raise AttributeError(f"Layer '{layer_name}' not found in the model.")

    def unfreeze_layers(self, layers_to_unfreeze):
        """
        Unfreeze specific layers.
        :param layers_to_unfreeze: List of layer names to unfreeze.
        """
        for layer_name in layers_to_unfreeze:
            if hasattr(self.model, layer_name):
                for param in getattr(self.model, layer_name).parameters():
                    param.requires_grad = True
                print(f"✅ Layer '{layer_name}' has been unfrozen.")
            else:
                raise AttributeError(f"Layer '{layer_name}' not found in the model.")


class TransferLearningModel(Architecture):
    """
    TransferLearningModel: Specialized class for models built from transfer learning.
    Handles fine-tuning, metadata tracking, and preparation for deployment as ProductionModel.
    """
    def __init__(self, model_name, base_models_info, architecture_class):
        """
        Initialize the TransferLearningModel.

        :param model_name: Name of the transfer learning model.
        :param base_models_info: List of dicts with base model info:
            [{"name": "ModelA", "layers_used": ["encoder"], "frozen": True}, ...]
        :param architecture_class: The architecture to use as a base (e.g., GRUseq2seq).
        """
        super().__init__(model_name=model_name)
        self.base_models_info = base_models_info  # Metadata about source models
        self.architecture_class = architecture_class
        self.training_info = {}  # To be updated during training

        # Dynamically initialize architecture
        self.model = self.architecture_class()

        self._load_base_models()

    def _load_base_models(self):
        """
        Load base models, extract the specified layers, and set their weights.
        """
        self.loaded_layers = {}

        for model_info in self.base_models_info:
            model_name = model_info['name']
            layers_used = model_info['layers_used']
            frozen = model_info.get('frozen', False)

            # Load each model via ProductionModel
            from NeuralLib.production_models.base import ProductionModel
            base_model = ProductionModel(
                model_name=model_name,
                hugging_repo=f"marianaagdias/{model_name}",
                architecture_class=self.architecture
            )

            for layer_name in layers_used:
                if not hasattr(base_model.model, layer_name):
                    raise AttributeError(f"Layer '{layer_name}' not found in model '{model_name}'")

                # Extract the layer
                layer = getattr(base_model.model, layer_name)
                self.loaded_layers[layer_name] = layer

                # Freeze if specified
                if frozen:
                    for param in layer.parameters():
                        param.requires_grad = False

                # Attach layer to the current model dynamically
                setattr(self, layer_name, layer)

        print("✅ Layers loaded and set successfully from base models.")

    def promote_to_production(self, results_directory):
        """Prepare the model for deployment as a ProductionModel."""
        weights_path = os.path.join(results_directory, 'model_weights.pth')
        torch.save(self.state_dict(), weights_path)
        print(f"Model weights saved at {weights_path}")

        hparams_path = os.path.join(results_directory, 'hparams.yaml')
        with open(hparams_path, 'w') as f:
            yaml.dump(self.hparams, f)
        print(f"Hyperparameters saved at {hparams_path}")

        print("✅ Transfer Learning Model successfully promoted to ProductionModel.")
