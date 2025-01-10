from NeuralLib.production_models.base import ProductionModel
from NeuralLib.transfer_learning.tlmodel import TLModel
import torch.nn as nn
import torch


class TLFactory:
    """
    Factory for configuring TransferLearningModel with weights from ProductionModels.
    """

    def __init__(self):
        self.models = {}

    def load_production_model(self, model_name, hugging_repo, architecture_class):
        """
        Load a production model and store it in the factory.
        """
        model = ProductionModel(
            model_name=model_name,
            hugging_repo=hugging_repo,
            architecture_class=architecture_class
        )
        self.models[model_name] = model
        print(f"✅ Loaded ProductionModel: {model_name}")

    def configure_tl_model(self, tl_model, layer_mapping, freeze_layers=None, unfreeze_layers=None):
        """
        Configure a TransferLearningModel by injecting weights and managing layer freezing/unfreezing.

        :param tl_model: Instance of TLModel to configure.
        :param layer_mapping: Dict mapping target layers in TLModel to source state_dicts.
                              Example: {'model.gru_layers.0': source_GRU_layer.state_dict()}
        :param freeze_layers: List of layer names to freeze.
        :param unfreeze_layers: List of layer names to unfreeze.
        """
        # Step 1: Inject Weights
        print("🔄 Injecting weights into TLModel layers...")
        tl_model.inject_weights(layer_mapping)

        # Step 2: Freeze Layers
        if freeze_layers:
            print("Freezing specified layers...")
            for layer_name in freeze_layers:
                try:
                    parts = layer_name.split('.')
                    current_attr = tl_model.model
                    for part in parts:
                        if part.isdigit():
                            current_attr = current_attr[int(part)]
                        else:
                            current_attr = getattr(current_attr, part)
                    for param in current_attr.parameters():
                        param.requires_grad = False
                    print(f"✅ Layer '{layer_name}' frozen.")
                except (AttributeError, IndexError) as e:
                    raise ValueError(f"Failed to freeze layer '{layer_name}': {e}")

        # Step 3: Unfreeze Layers
        if unfreeze_layers:
            print("Unfreezing specified layers...")
            for layer_name in unfreeze_layers:
                try:
                    parts = layer_name.split('.')
                    current_attr = tl_model.model
                    for part in parts:
                        if part.isdigit():
                            current_attr = current_attr[int(part)]
                        else:
                            current_attr = getattr(current_attr, part)
                    for param in current_attr.parameters():
                        param.requires_grad = True
                    print(f"✅ Layer '{layer_name}' unfrozen.")
                except (AttributeError, IndexError) as e:
                    raise ValueError(f"Failed to unfreeze layer '{layer_name}': {e}")

        print("✅ TLModel successfully configured!")

    def configure_tl_model_old(self, tl_model, layer_mapping, freeze_layers=None, unfreeze_layers=None):
        """
        Configure a TransferLearningModel with layer mappings and freezing strategies.
        :param tl_model: An instance of TransferLearningModel.
        :param layer_mapping: Dict mapping TLModel layers to ProductionModel layers.
        :param freeze_layers: List of layers to freeze.
        :param unfreeze_layers: List of layers to unfreeze.
        """
        # Map weights
        mapped_layers = {}
        for tl_layer, mapping_info in layer_mapping.items():
            source_model = self.models.get(mapping_info['source_model'])
            if not source_model:
                raise ValueError(f"Production model '{mapping_info['source_model']}' not loaded.")
            source_layer = getattr(source_model.model, mapping_info['source_layer'])
            mapped_layers[tl_layer] = source_layer

        tl_model.inject_weights(mapped_layers)

        # Freeze layers
        if freeze_layers:
            tl_model.freeze_layers(freeze_layers)

        # Unfreeze layers
        if unfreeze_layers:
            tl_model.unfreeze_layers(unfreeze_layers)

        print("✅ TransferLearningModel successfully configured.")


class TransferLearningFactory:
    """
    Factory for creating TransferLearningModel instances using multiple pretrained models.
    """

    def __init__(self):
        self.models = {}

    def load_production_model(self, model_name, hugging_repo, architecture_class):
        """
        Load an existing production model.
        """
        model = ProductionModel(
            model_name=model_name,
            hugging_repo=hugging_repo,
            architecture_class=architecture_class
        )
        self.models[model_name] = model
        print(f"✅ Model '{model_name}' loaded successfully.")
        return model

    def create_transfer_learning_model(self, model_name, base_models_info, architecture_class):
        """
        Create a TransferLearningModel using loaded production models.

        :param model_name: Name of the new transfer learning model.
        :param base_models_info: List of dictionaries specifying layers and freeze status.
        :param architecture_class: Base architecture for the new model.
        :return: TransferLearningModel instance.
        """
        for model_info in base_models_info:
            if model_info['name'] not in self.models:
                raise ValueError(f"Model {model_info['name']} not loaded in factory. Call load_production_model first.")

        # Create the TransferLearningModel
        transfer_model = TLModel(
            model_name=model_name,
            base_models_info=base_models_info,
            architecture_class=architecture_class
        )
        print(f"✅ Created TransferLearningModel: {model_name}")
        return transfer_model


