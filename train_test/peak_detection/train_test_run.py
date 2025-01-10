import data_preprocessing.gib01 as gib
from config import RESULTS_PEAK_DETECTION
import architectures as a
from production_models.test_models_to_remove_ import test_peak_detection_test_set
import time
import os
import glob
import torch

RUN_MODE = 'test'  # Options: 'train', 'test', 'train_and_test'
# if RUN_MODE == 'test', define the checkpoints directory of the model
CHECKPOINTS_DIRECTORY = os.path.join(RESULTS_PEAK_DETECTION, 'checkpoints',
                                     'GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26')
ALL_SAMPLES_TEST = False
SAMPLES_TEST = 3

# Default hyperparameters
N_FEATURES = 1
HID_DIM = 64
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.3
EPOCHS = 80
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # for training (for testing it is always 1)
GPU_ID = 0
TASK = 'classification'
NUM_CLASSES = 1
PATIENCE = 20
THRESHOLD = 0.5


# 1. Train Model
def train(architecture=a.GRUseq2seq, hid_dim=HID_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
          gpu_id=GPU_ID, epochs=EPOCHS, patience=PATIENCE,
          enable_tensorboard=True, results_directory=RESULTS_PEAK_DETECTION):
    model = architecture(
        n_features=N_FEATURES,
        hid_dim=hid_dim,
        n_layers=n_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        bidirectional=BIDIRECTIONAL,
        task=TASK,
        num_classes=NUM_CLASSES
    )

    start_time = time.time()
    model.train_model(
        path_x=gib.X,
        path_y=gib.Y_BIN,
        all_samples=True,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        dataset_name='gib01',
        trained_for='peak detection',
        enable_tensorboard=enable_tensorboard,
        gpu_id=gpu_id,
        results_directory=results_directory
    )

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
    return model.checkpoints_directory


# 2. Test Model
def test(checkpoints_dir, threshold=THRESHOLD, results_directory=RESULTS_PEAK_DETECTION):
    ckpt_file = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))[0]

    test_peak_detection_test_set(
        model_checkpoint=ckpt_file,
        path_x=gib.X,
        path_y=gib.Y_BIN,
        n_features=N_FEATURES,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        threshold=threshold,
        all_samples=ALL_SAMPLES_TEST,
        samples=SAMPLES_TEST
    )
    print(f"Testing completed using checkpoint from {checkpoints_dir}")


# 3. Train and Test Model
def train_and_test(**kwargs):
    checkpoints_dir = train(**kwargs)
    test(checkpoints_dir=checkpoints_dir)


# 4. Run Model on a Single Signal
def run_on_single_signal(checkpoints_dir, signal, threshold=THRESHOLD):
    # Load model and checkpoint
    ckpt_file = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))[0]
    model = a.GRUseq2seq.load_from_checkpoint(
        ckpt_file,
        n_features=N_FEATURES,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )

    # Prepare signal for testing
    model.eval()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    signal = torch.tensor(signal).float().unsqueeze(0)  # Add batch dimension
    lengths = [signal.size(1)]

    # Run inference
    with torch.no_grad():
        output = model(signal, lengths)
        output_probs = torch.sigmoid(output).squeeze()
        output_binary = (output_probs > threshold).float()
        peak_indices = torch.nonzero(output_binary).squeeze().cpu().numpy()

    print(f"Peak indices for the provided signal: {peak_indices}")
    return peak_indices


# Main function to control the flow
if __name__ == '__main__':
    run_mode = RUN_MODE

    if run_mode == 'train':
        # Train only
        train()

    elif run_mode == 'test':
        # Test only, requires the checkpoints directory
        checkpoints_directory = CHECKPOINTS_DIRECTORY
        test(checkpoints_directory)

    elif run_mode == 'train_and_test':
        # Train and test in one go
        train_and_test()
