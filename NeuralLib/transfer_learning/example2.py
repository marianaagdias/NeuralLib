from NeuralLib.transfer_learning import TLModel, TLFactory
from NeuralLib.architectures import GRUseq2seq
from NeuralLib.data_preprocessing.gib01 import X, Y_BIN
from NeuralLib.config import RESULTS_PEAK_FINE_TUNING

# Initialize factory
factory = TLFactory()

# Load Production Model
factory.load_production_model(model_name="ECGPeakDetector",
                              hugging_repo='marianaagdias/ecg_peak_detection',
                              architecture_class=GRUseq2seq)

prod_model = factory.models['ECGPeakDetector']

# Inspecting GRUseq2seq model structure
# print(prod_model.model)

# Define hyperparameters for the TLModel
arch_params = {
    'n_features': 1,
    'hid_dim': [32, 64, 64],  # Hidden dimensions per layer
    'n_layers': 3,            # Total layers
    'dropout': 0,             # Dropout rate
    'learning_rate': 0.001,
    'bidirectional': True,
    'task': 'classification',
    'num_classes': 1
}

# Initalize TLModel
tl_model = TLModel(GRUseq2seq, **arch_params)

# Print all layer names and their types
for name, module in tl_model.named_modules():
    print(name, "->", module)

# Print all keys in the state_dict
for key in tl_model.state_dict().keys():
    print(key)

# Extract weights from the Production Model
layer_mapping = {
    'gru_layers.0': prod_model.model.gru_layers[0].state_dict(),  # First GRU layer weights
    'gru_layers.1': prod_model.model.gru_layers[1].state_dict()   # Second GRU layer weights
}

# Define freezing and unfreezing strategies
freeze_layers = ['gru_layers.0']
unfreeze_layers = ['gru_layers.1']

# Configure the TLModel in the factory
factory.configure_tl_model(
    tl_model=tl_model,
    layer_mapping=layer_mapping,
    freeze_layers=freeze_layers,
    unfreeze_layers=unfreeze_layers
)

# define training parameters for training TLModel
train_params = {
    'path_x': X,
    'path_y': Y_BIN,
    'epochs': 1,
    'batch_size': 1,
    'patience': 2,
    'dataset_name': 'private_gib01',
    'trained_for': 'fine-tuning peak detection',
    'all_samples': False,
    'samples': 3,
    'gpu_id': None,
    'results_directory': RESULTS_PEAK_FINE_TUNING,
    'enable_tensorboard': True
}

# Train TLModel
tl_model.train_tl(**train_params)

