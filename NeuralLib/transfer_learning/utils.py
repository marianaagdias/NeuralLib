import torch.nn as nn


def extract_feature_extractor(model):
    """
    Extract feature extractor layers from a given model.
    """
    if hasattr(model, 'feature_extractor'):
        return model.feature_extractor
    raise AttributeError("Model does not have a 'feature_extractor' attribute.")


def adapt_encoder_to_new_task(encoder, num_classes):
    """
    Adapt an encoder to a new task by appending a new classifier head.
    """

    return nn.Sequential(
        encoder,
        nn.Linear(encoder.output_dim, num_classes)
    )


def freeze_all_except(model, unfrozen_layers):
    """
    Freeze all layers except specified ones.
    """
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in unfrozen_layers)

