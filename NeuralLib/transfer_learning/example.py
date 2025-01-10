from NeuralLib.transfer_learning.factory import TLFactory
from NeuralLib.transfer_learning.tlmodel import TLModel
from NeuralLib.architectures import GRUseq2seq

# Initialize factory
factory = TLFactory()

# Load Production Models
factory.load_production_model("ProdModelA", "user/ProdModelA", GRUseq2seq)
factory.load_production_model("ProdModelB", "user/ProdModelB", GRUseq2seq)

# Define architecture hyperparameters
arch_params = {
    'n_features': 1,
    'hid_dim': 16,
    'n_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.01,
    'bidirectional': True,
    'task': 'classification',
    'num_classes': 1,
}

# Create TransferLearningModel
tl_model = TLModel(GRUseq2seq, **arch_params)

# Define layer mapping and freezing strategy
layer_mapping = {
    'encoder_layer1': {'source_model': 'ProdModelA', 'source_layer': 'encoder'},
    'decoder_layer2': {'source_model': 'ProdModelB', 'source_layer': 'decoder'}
}
freeze_layers = ['encoder_layer1']
unfreeze_layers = ['decoder_layer2']

# Configure TLModel
factory.configure_tl_model(
    tl_model=tl_model,
    layer_mapping=layer_mapping,
    freeze_layers=freeze_layers,
    unfreeze_layers=unfreeze_layers
)

# Retrain TLModel
tl_model.train_tl(
    path_x="data/x_train.npy",
    path_y="data/y_train.npy",
    patience=5,
    batch_size=32,
    epochs=20,
    results_directory="results/transfer_ecg_model",
    dataset_name="ECG Dataset",
    trained_for="ECG Fine-Tuning"
)

