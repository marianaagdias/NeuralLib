import torch
import pytest
from NeuralLib.architectures.biosignals_architectures import TransformerEncoderDecoder


@pytest.fixture
def model():
    return TransformerEncoderDecoder(
        model_name="test_transformerencdec",
        n_features=5,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=32,
        dropout=0.2,
        learning_rate=0.001
    )


def test_transformerencoderdecoder_forward(model):
    batch_size, src_len, tgt_len = 4, 20, 15
    src_tensor = torch.randn(batch_size, src_len, model.n_features)
    tgt_tensor = torch.randn(batch_size, tgt_len, model.n_features)

    output = model(src_tensor, tgt_tensor)

    assert output.shape == (batch_size, tgt_len, model.n_features), \
        f"Expected {(batch_size, tgt_len, model.n_features)}, got {output.shape}"
