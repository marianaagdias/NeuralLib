from NeuralLib.transfer_learning.factory import TransferLearningFactory
from NeuralLib.architectures import GRUseq2seq
import torch.optim as optim
import torch
# Example dataloader (dummy)
from torch.utils.data import DataLoader, TensorDataset

# Initialize the factory
factory = TransferLearningFactory()

# Load a production model
model = factory.load_production_model(
    model_name="ECGPeakDetector",
    hugging_repo="marianaagdias/ecg_peak_detection",
    architecture_class=GRUseq2seq
)
# Extract the encoder
encoder = factory.extract_encoder(model)

# Build a custom model for a new task
custom_model = factory.build_custom_model(encoder, num_classes=5)

# Freeze specific layers
custom_model2 = factory.freeze_layers(custom_model, layers_to_freeze=['encoder.layer1', 'encoder.layer2'])

optimizer = optim.Adam(custom_model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

X_train = torch.rand(100, 10)  # Example tensor
y_train = torch.randint(0, 5, (100,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8)

# Fine-tune
fine_tuned_model = factory.fine_tune_model(custom_model, train_loader, optimizer, loss_fn, epochs=5)

