import torch
import pytest
from NeuralLib.architectures.biosignals_architectures import Transformerseq2seq


def test_transformerseq2seq_forward():
    """Unit test for forward pass of Transformerseq2seq model."""

    # Define model parameters
    model_name = "TestTransformer"
    n_features = 5  # Input/output features per timestep
    d_model = 16  # Embedding dimension for transformer
    nhead = 2  # Number of attention heads
    num_encoder_layers = 2  # Encoder depth
    num_decoder_layers = 2  # Decoder depth
    dim_feedforward = 64  # Feedforward layer dimension
    dropout = 0.1  # Dropout probability
    learning_rate = 0.001  # Learning rate

    # Instantiate model
    model = Transformerseq2seq(
        model_name=model_name,
        n_features=n_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        learning_rate=learning_rate
    )

    # Define batch size and sequence length
    batch_size, seq_len = 4, 20

    # Generate random input tensors (shape: [batch_size, seq_len, n_features])
    src_tensor = torch.randn(batch_size, seq_len, n_features)
    tgt_tensor = torch.randn(batch_size, seq_len, n_features)

    # Forward pass
    output = model(src_tensor, tgt_tensor)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, n_features), f"Unexpected output shape: {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__])
