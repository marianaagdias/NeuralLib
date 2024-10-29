import itertools
from train_test_run import train  # Import the train function

# Define hyperparameter grid
HID_DIM_OPTIONS = [32, 64]  # 128 , 256]
N_LAYERS_OPTIONS = [2]  # , 3]
DROPOUT_OPTIONS = [0.0, 0.3]  # , 0.5]
# LEARNING_RATE_OPTIONS = [0.001, 0.0001]
# BATCH_SIZE_OPTIONS = [32, 64]
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 3


# 2. Grid Search over the hyperparameter combinations
def run_grid_search():
    combinations = list(itertools.product(HID_DIM_OPTIONS, N_LAYERS_OPTIONS, DROPOUT_OPTIONS))

    for i, (hid_dim, n_layers, dropout) in enumerate(combinations, 1):
        print(f"\nTraining model {i}/{len(combinations)} with parameters:")
        print(
            f"  hid_dim={hid_dim}, n_layers={n_layers}, dropout={dropout}")

        # Call the imported train function with the current set of hyperparameters
        checkpoints_dir = train(
            hid_dim=hid_dim,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )

        print(f"Model {i} completed. Checkpoints saved to: {checkpoints_dir}")


# Main function to run grid search
if __name__ == '__main__':
    run_grid_search()
