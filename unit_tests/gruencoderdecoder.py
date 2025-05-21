import torch
import pytest
from NeuralLib.architectures.biosignals_architectures import GRUED


@pytest.fixture
def model():
    """Fixture to initialize a test instance of the GRUEncoderDecoder model."""
    return GRUED(
        model_name="test_model",
        n_features=5,
        enc_hid_dim=10,  # Encoder hidden size
        dec_hid_dim=20,  # Decoder hidden size (should match hidden_dim * 2 if concatenated)
        enc_layers=2,
        dec_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        bidirectional=True
    )


def test_gruencoderdecoder_forward(model):
    batch_size, src_len, tgt_len = 4, 20, 15
    src_tensor = torch.randn(batch_size, src_len, model.n_features)
    tgt_tensor = torch.randn(batch_size, tgt_len, model.n_features)
    src_lengths = torch.tensor([src_len] * batch_size)
    tgt_lengths = torch.tensor([tgt_len] * batch_size)

    output = model(src_tensor, tgt_tensor, src_lengths, tgt_lengths)

    # Assert the output shape matches expected dimensions
    assert output.shape == (batch_size, tgt_len, model.n_features), \
        f"Expected output shape {(batch_size, tgt_len, model.n_features)}, but got {output.shape}"
