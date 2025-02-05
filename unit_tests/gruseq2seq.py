import torch
import pytest
from NeuralLib.architectures.biosignals_architectures import GRUseq2seq


@pytest.fixture
def model():
    return GRUseq2seq(
        model_name="test_seq2seq",
        n_features=5,
        hid_dim=10,
        n_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        bidirectional=True,
        task="classification",
        num_classes=3,
        multi_label=False
    )


def test_gruseq2seq_forward(model):
    batch_size, seq_len = 4, 20
    input_tensor = torch.randn(batch_size, seq_len, model.n_features)
    lengths = torch.tensor([seq_len] * batch_size)

    output = model(input_tensor, lengths)

    assert output.shape == (batch_size, seq_len, model.num_classes), \
        f"Expected {(batch_size, seq_len, model.num_classes)}, got {output.shape}"
