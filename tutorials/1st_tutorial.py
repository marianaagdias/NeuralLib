from architectures import GRUseq2seq
from data_preprocessing.gib01 import X, Y_BIN
from config import RESULTS_PEAK_DETECTION

# Step 1: Define architecture parameters
arch_params = {
    'n_features': 1,
    'hid_dim': 16,
    'n_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.01,
    'bidirectional': True,
    'task': 'classification',
    'num_classes': 1,
}

# Step 2: Define training parameters
train_params_ = {
    'path_x': X,
    'path_y': Y_BIN,
    'epochs': 3,
    'batch_size': 1,
    'patience': 2,
    'dataset_name': 'private_gib01',
    'trained_for': 'peak detection',
    'all_samples': False,
    'samples': 3,
    'gpu_id': None,
    'results_directory': RESULTS_PEAK_DETECTION,
    'enable_tensorboard': True
}

# Step 3: Initialize and train the GRUseq2seq model
print("Training from scratch...")
model = GRUseq2seq(**arch_params)
model.train_from_scratch(
    path_x=train_params_['path_x'],
    path_y=train_params_['path_y'],
    patience=train_params_['patience'],
    batch_size=train_params_['batch_size'],
    epochs=train_params_['epochs'],
    results_directory=train_params_['results_directory'],
    gpu_id=train_params_['gpu_id'],
    all_samples=train_params_['all_samples'],
    samples=train_params_['samples'],
    dataset_name=train_params_['dataset_name'],
    trained_for=train_params_['trained_for'],
    enable_tensorboard=train_params_['enable_tensorboard']
)

# Save checkpoints directory after initial training
checkpoints_dir = model.checkpoints_directory

# Step 4: Retrain the model for 2 more epochs
print("Retraining...")
train_params_retrain = train_params_.copy()
train_params_retrain['epochs'] = 2
model.retrain(
    path_x=train_params_retrain['path_x'],
    path_y=train_params_retrain['path_y'],
    patience=train_params_retrain['patience'],
    batch_size=train_params_retrain['batch_size'],
    epochs=train_params_retrain['epochs'],
    results_directory=train_params_retrain['results_directory'],
    gpu_id=train_params_retrain['gpu_id'],
    all_samples=train_params_retrain['all_samples'],
    samples=train_params_retrain['samples'],
    dataset_name=train_params_retrain['dataset_name'],
    trained_for=train_params_retrain['trained_for'],
    checkpoints_directory=checkpoints_dir,
    enable_tensorboard=train_params_retrain['enable_tensorboard'],
)

# Step 5: Test the model on the test set
print("Testing on test set...")
predictions, avg_loss = model.test_on_test_set(
    path_x=train_params_['path_x'],
    path_y=train_params_['path_y'],
    checkpoints_dir=checkpoints_dir,
    gpu_id=train_params_['gpu_id'],
    all_samples=False, # if True, test on all available samples
    samples=5,
    save_predictions=True
)

print(f"Average Test Loss: {avg_loss:.4f}")
