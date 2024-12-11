import os.path

import architectures as arc
from config import RESULTS_PEAK_DETECTION
from data_preprocessing.gib01 import X, Y_BIN
from production_models import post_process_peaks_binary

# Define architecture and training parameters
architecture_name = 'GRUseq2seq'
# print(arc.get_valid_architectures())
archi_params_options = {
    "n_features": [1],
    "hid_dim": [[32, 64, 64], [64, 64, 64], [64, 128, 64]],
    "n_layers": [3],
    "dropout": [0.3, 0],
    "learning_rate": [0.001, 0.0005],
    "bidirectional": [True],
    "task": ["classification"],
    "num_classes": [1],
}

train_params = {
    'path_x': X,
    'path_y': Y_BIN,
    'epochs': 200,
    'batch_size': 32,
    'patience': 20,
    'dataset_name': 'private_gib01',
    'trained_for': 'peak detection',
    'all_samples': True,
    'gpu_id': 0,
    'results_directory': RESULTS_PEAK_DETECTION,
    'enable_tensorboard': True,
}

# Perform grid search
print("Performing grid search...")
best_dir, best_val_loss, val_losses = arc.run_grid_search(
    architecture_name, archi_params_options, train_params)
print(f"Best model found with validation loss: {best_val_loss:.4f}, saved in {best_dir}")

# Step 4: Test the best model on the test set
print("Testing the best model on the test set...")
# 4.1. Load architecture parameters from the hparams.yaml file
architecture_params = arc.get_hparams_from_checkpoints(best_dir)
# 4.2 Initialize the model using the loaded parameters
model = arc.GRUseq2seq(**architecture_params)

predictions, avg_loss = model.test_on_test_set(
    path_x=train_params["path_x"],
    path_y=train_params["path_y"],
    checkpoints_dir=best_dir,
    gpu_id=train_params["gpu_id"],
    save_predictions=True,
    all_samples=True,
    # samples=5,
    post_process_fn=lambda output: post_process_peaks_binary(output, threshold=0.5, filter_peaks=True)
)
print(f"Testing complete. Average Test Loss: {avg_loss:.4f}")

