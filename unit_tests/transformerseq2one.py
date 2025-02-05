import torch
import pytest
from NeuralLib.architectures.biosignals_architectures import Transformerseq2one


@pytest.fixture
def model():
    return Transformerseq2one(
        model_name="test_transformerseq2one",
        n_features=5,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=32,
        dropout=0.2,
        learning_rate=0.001,
        num_classes=3
    )


def test_transformerseq2one_forward(model):
    batch_size, seq_len = 4, 20
    input_tensor = torch.randn(batch_size, seq_len, model.n_features)

    output = model(input_tensor)

    assert output.shape == (batch_size, model.num_classes), \
        f"Expected {(batch_size, model.num_classes)}, got {output.shape}"
