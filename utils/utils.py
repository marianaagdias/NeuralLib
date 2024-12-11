import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def configure_device(gpu_id=None):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)} (GPU ID: {gpu_id})")
        return gpu_id
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)  # Default to the first GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (GPU ID: 0)")
        return 0
    else:
        print("No GPU available, using CPU.")
        return 'cpu'


def list_gpus():
    """ List the names of available gpus. """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU ID: {i}, Model: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available.")


# é preciso handle o facto de estarmos a ver ponto a ponto do sinal e nao idx a idx do dataset
def calculate_class_weights(dataset):
    '''
    ALTERAR!
    :param dataset:
    :return:
    '''
    labels = []
    for _, label in dataset:
        labels.append(label)

    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()

    return class_weights


def save_model_results(model, results_dir, model_name, best_val_loss):
    checkpoints_dir = os.path.join(results_dir, 'checkpoints', model_name)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Save model hyperparameters and results
    results = {
        'hyperparameters': {
            'n_features': model.n_features,
            'hid_dim': model.hid_dim,
            'n_layers': model.n_layers,
            'dropout': model.dropout,
            'learning_rate': model.learning_rate,
            'bidirectional': model.bidirectional,
        },
        'best_validation_loss': best_val_loss,
        'best_epoch': model.trainer.current_epoch
    }

    results_file = os.path.join(checkpoints_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)


def save_predictions(predictions, batch_idx, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_path = os.path.join(dir, f"predictions_batch_{batch_idx}.npy")
    np.save(file_path, np.array(predictions))


def save_predictions_with_filename(predictions, input_filename, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Use the input filename to name the prediction file
    file_name_without_ext = os.path.splitext(input_filename)[0]
    file_path = os.path.join(dir, f"{file_name_without_ext}.npy")

    # Save the predictions
    np.save(file_path, np.array(predictions))


def collate_fn(batch):
    # Separate inputs (X) and labels (Y)
    X = [item[0] for item in batch]  # List of inputs
    Y = [item[1] for item in batch]  # List of labels

    # Sort by the length of X in descending order
    X_lengths = [len(x) for x in X]
    sorted_indices = sorted(range(len(X_lengths)), key=lambda i: X_lengths[i], reverse=True)

    X = [X[i] for i in sorted_indices]
    Y = [Y[i] for i in sorted_indices]

    # Pad X and Y
    X_padded = pad_sequence(X, batch_first=True)
    Y_padded = pad_sequence(Y, batch_first=True)

    # Return both padded sequences and their lengths
    return X_padded, Y_padded, X_lengths


class DatasetSequence(Dataset):

    def __init__(self, path_x, path_y, part='train', all_samples=False, samples=None, features=1, overlap=None):
        self.dir_x = os.path.join(path_x, part)
        self.dir_y = os.path.join(path_y, part)
        self.all_samples = all_samples
        self.samples = samples
        self.features = features  # still to be done
        self.overlap = overlap  # still to be done

        # Check if directories exist
        if not os.path.isdir(self.dir_x):
            print(f"Error: Directory {self.dir_x} does not exist.")
        if not os.path.isdir(self.dir_y):
            print(f"Error: Directory {self.dir_y} does not exist.")

        # Check if there are any .npy files in the directories
        self.files_x = [f for f in os.listdir(self.dir_x) if f.endswith('.npy')]
        self.files_y = [f for f in os.listdir(self.dir_y) if f.endswith('.npy')]
        # print(self.files_x)
        # print(self.files_y)

        if len(self.files_x) == 0:
            print(f"Error: No .npy files found in {self.dir_x}.")
        if len(self.files_y) == 0:
            print(f"Error: No .npy files found in {self.dir_y}.")

        # Check if samples is an integer when all_samples is False
        if not self.all_samples:
            if not isinstance(self.samples, int):
                raise ValueError("Error: The number of samples to be used should be provided.")
            elif self.samples > len(self.files_x):
                print(f"Warning: Requested {self.samples} samples, but only {len(self.files_x)} files are available.")
                self.samples = len(self.files_x)  # Adjust to the maximum available

    def __len__(self):
        if self.all_samples:
            print('Using all data samples')
            return len(self.files_x)
        else:
            # print(f"Using {min(self.samples, len(self.files_x))} data samples")
            return min(self.samples, len(self.files_x))

    def __getitem__(self, idx):
        if idx >= len(self.files_x):
            print(f"Error: Index {idx} is out of bounds for the dataset with {len(self.files_x)} samples.")
            return None

        # Load the data
        x_path = os.path.join(self.dir_x, self.files_x[idx])
        y_path = os.path.join(self.dir_y, self.files_x[idx])  # files_x is correct. to make sure it is loading the file
        # with the same name for x and y.

        try:
            item_x = np.load(x_path)
        except Exception as e:
            print(f"Error loading file {x_path}: {e}")
            return None

        try:
            item_y = np.load(y_path)
        except Exception as e:
            print(f"Error loading file {y_path}: {e}")
            return None

        # print(f"x shape: {item_x.shape}")
        # print(f"y shape: {item_y.shape}")

        item_x = item_x.reshape(-1, 1)  # Reshape to (seq_len, 1)
        item_y = item_y.reshape(-1, 1)  # Reshape to (seq_len, 1)

        return torch.tensor(item_x).float(), torch.tensor(item_y).float()

