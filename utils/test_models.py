import os
import torch
from torch.utils.data import DataLoader
from BaseModel.model_architectures import GRUseq2seq
from utils import DatasetSequence, save_predictions_with_filename
import numpy as np
import BaseModel.model_architectures as bm
from config import RESULTS_PEAK_DETECTION
import glob


def test_peak_detection_test_set(model_checkpoint, path_x, path_y, n_features, hid_dim, n_layers, dropout,
                                 save_preds=True, threshold=0.5, all_samples=True, samples=None,
                                 device=None):
    # Load the test dataset
    test_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='test', all_samples=all_samples, samples=samples)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # I am forcing the batch size to be one

    # Load the trained model from the checkpoint
    model = GRUseq2seq.load_from_checkpoint(
        model_checkpoint,
        n_features=n_features,
        hid_dim=hid_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    # Move the model to the appropriate device (GPU or CPU)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    # Create a directory to save predictions
    checkpoints_dir = os.path.dirname(model_checkpoint)
    predictions_dir = os.path.join(checkpoints_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Run inference and process the outputs
    for batch_idx, (X, Y) in enumerate(test_dataloader):
        X = X.to(device)
        Y = Y.to(device)

        lengths = [X.size(1)]  # X is just one signal because batch_size=1

        output = model(X, lengths)
        loss = model.criterion(output, Y)

        model.log('test_loss', loss)
        print(f"Batch {batch_idx}: Test Loss: {loss.item()}")

        # Convert output to binary predictions and extract peak indices
        output_probs = torch.sigmoid(output).squeeze()
        output_binary = (output_probs > threshold).float()
        # output_binary = (torch.sigmoid(output) > threshold).float()
        # print(output_binary)
        all_peak_indices = torch.nonzero(output_binary).squeeze()
        # print(all_peak_indices)

        # Ensure all_peak_indices is a 1D array or handle empty cases
        if all_peak_indices.numel() == 0:  # No peaks found
            filtered_peak_indices = np.array([])  # Empty array
            # print(filtered_peak_indices)
        else:
            all_peak_indices = all_peak_indices.view(-1).cpu().numpy()  # Convert to numpy if non-empty
            # print(all_peak_indices)

            # Remove extra peaks
            if len(all_peak_indices) > 1:
                peak_differences = np.diff(all_peak_indices)  # Calculate differences between consecutive peaks
                if np.any(peak_differences < 40):  # Check if any peaks are closer than 40 samples
                    # Perform non-maximum suppression (only if necessary) by iterating through the peaks
                    filtered_peak_indices = []
                    i = 0
                    while i < len(all_peak_indices):
                        # Define the current window: 40 samples ahead from the current peak
                        window_start = all_peak_indices[i]
                        window_end = window_start + 40
                        # Find all peaks within the current window
                        window_peaks = all_peak_indices[(all_peak_indices >= window_start) & (all_peak_indices < window_end)]
                        # Keep the peak with the highest probability in this window
                        if len(window_peaks) > 0:
                            max_peak = window_peaks[np.argmax(output_probs[window_peaks].detach().cpu().numpy())]
                            filtered_peak_indices.append(max_peak)
                        # Move the index to the next window (after the current window's end)
                        i += len(window_peaks)
                    # Convert filtered_peak_indices to numpy array for saving
                    filtered_peak_indices = np.array(filtered_peak_indices)
                    # Find
                else:
                    filtered_peak_indices = all_peak_indices
            else:
                filtered_peak_indices = all_peak_indices

        # Save the predictions
        if save_preds:
            input_file_name = os.path.basename(test_dataset.files_x[batch_idx])  # Get the original file name
            save_predictions_with_filename(filtered_peak_indices, input_file_name, predictions_dir)

    print(f"Predictions saved at: {predictions_dir}")


# Run peak detection model on a single signal - testar
def peak_detection(signal, checkpoints_dir=os.path.join(RESULTS_PEAK_DETECTION, 'checkpoints',
                                                        'GRUseq2seq_64hid_3l_lr0.001_drop0.3_dt2024-10-18_18-01-26'),
                   threshold=0.5, min_distance=40):
    # Load model and checkpoint
    ckpt_file = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))[0]
    model = bm.GRUseq2seq.load_from_checkpoint(
        ckpt_file,
        n_features=1,
        hid_dim=64,
        n_layers=3,
        dropout=0.3
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
        all_peak_indices = torch.nonzero(output_binary).squeeze().cpu().numpy()

        # Ensure all_peak_indices is a 1D array or handle empty cases
        if all_peak_indices.numel() == 0:  # No peaks found
            filtered_peak_indices = np.array([])  # Empty array
            # print(filtered_peak_indices)
        else:
            all_peak_indices = all_peak_indices.view(-1).cpu().numpy()  # Convert to numpy if non-empty
            # print(all_peak_indices)

            # Remove extra peaks
            if len(all_peak_indices) > 1:
                peak_differences = np.diff(all_peak_indices)  # Calculate differences between consecutive peaks
                if np.any(peak_differences < min_distance):  # Check if any peaks are closer than min_distance samples
                    # Perform non-maximum suppression (only if necessary) by iterating through the peaks
                    filtered_peak_indices = []
                    i = 0
                    while i < len(all_peak_indices):
                        # Define the current window: min_distance samples ahead from the current peak
                        window_start = all_peak_indices[i]
                        window_end = window_start + min_distance
                        # Find all peaks within the current window
                        window_peaks = all_peak_indices[
                            (all_peak_indices >= window_start) & (all_peak_indices < window_end)]
                        # Keep the peak with the highest probability in this window
                        if len(window_peaks) > 0:
                            max_peak = window_peaks[np.argmax(output_probs[window_peaks].detach().cpu().numpy())]
                            filtered_peak_indices.append(max_peak)
                        # Move the index to the next window (after the current window's end)
                        i += len(window_peaks)
                    # Convert filtered_peak_indices to numpy array for saving
                    filtered_peak_indices = np.array(filtered_peak_indices)
                    # Find
                else:
                    filtered_peak_indices = all_peak_indices
            else:
                filtered_peak_indices = all_peak_indices

    print(f"Peak indices for the provided signal: {filtered_peak_indices}")
    return filtered_peak_indices
