from NeuralLib.production_models.base import ProductionModel
import torch.nn as nn
import torch


class TransferLearningFactory:
    """
    Factory class for handling transfer learning workflows.
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

    def extract_encoder(self, model):
        """
        Extract the encoder (e.g., feature extractor) from a model.
        """
        if hasattr(model.model, "encoder"):
            return model.model.encoder
        else:
            raise AttributeError("The model does not have an 'encoder' attribute.")

    def extract_layer(self, model, layer_name):
        """
        Extract a specific layer by name.
        """
        if hasattr(model.model, layer_name):
            return getattr(model.model, layer_name)
        else:
            raise AttributeError(f"Layer '{layer_name}' not found in the model.")

    def freeze_layers(self, model, layers_to_freeze):
        """
        Freeze specific layers in the model.
        """
        for name, param in model.model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                print(f"❄️ Layer '{name}' frozen.")
        return model

    def build_custom_model(self, encoder, num_classes):
        """
        Build a custom model using the extracted encoder.
        """
        class CustomModel(nn.Module):
            def __init__(self, encoder, num_classes):
                super().__init__()
                self.encoder = encoder
                self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

            def forward(self, x):
                features = self.encoder(x)
                return self.classifier(features)

        print("✅ Custom model built successfully.")
        return CustomModel(encoder, num_classes)

    def fine_tune_model(self, model, dataloader, optimizer, loss_fn, epochs):
        """
        Fine-tune a model with new data.
        """
        model.train()
        for epoch in range(epochs):
            for X, y in dataloader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
            print(f"📈 Epoch {epoch+1}/{epochs} complete.")
        print("✅ Fine-tuning complete.")
        return model
