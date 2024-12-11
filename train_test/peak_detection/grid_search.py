import itertools
from train_test_run import train  # Import the train function


# 2. Grid Search over the hyperparameter combinations
def run_grid_search(architecture, hid_dim_op, n_layers_op, dropout_op, learning_rate_op, batch_size, epochs):
    combinations = list(itertools.product(hid_dim_op, n_layers_op, dropout_op, learning_rate_op))

    for i, (hid_dim, n_layers, dropout, learning_rate) in enumerate(combinations, 1):
        print(f"\nTraining model {i}/{len(combinations)} with parameters:")
        print(
            f"  hid_dim={hid_dim}, n_layers={n_layers}, dropout={dropout}, learning_rate ={learning_rate}")

        # Call the imported train function with the current set of hyperparameters
        checkpoints_dir = train(
            architecture=architecture,
            hid_dim=hid_dim,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs
        )

        print(f"Model {i} completed. Checkpoints saved to: {checkpoints_dir}")


# Main function to run grid search
if __name__ == '__main__':
    # Define hyperparameter grid
    HID_DIM_OPTIONS = [32, 64]  # 128 , 256]
    N_LAYERS_OPTIONS = [2]  # , 3]
    DROPOUT_OPTIONS = [0.0, 0.3]  # , 0.5]
    LEARNING_RATE_OPTIONS = [0.001]
    BATCH_SIZE = 32
    EPOCHS = 3
    run_grid_search(
        hid_dim_op=HID_DIM_OPTIONS,
        n_layers_op=N_LAYERS_OPTIONS,
        dropout_op=DROPOUT_OPTIONS,
        learning_rate_op=LEARNING_RATE_OPTIONS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
