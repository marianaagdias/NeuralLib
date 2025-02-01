import NeuralLib.architectures as arc
from NeuralLib.config import HUGGING_MODELS_BASE_DIR
from NeuralLib.utils import configure_device
import torch
import json
import os
import yaml
import numpy as np
from huggingface_hub import snapshot_download, get_collection


class ProductionModel(arc.Architecture):
    """
    Production Models
    A class for trained models, extending Architecture by adding weights and training information.
    Includes methods for importing from Hugging Face.

    Models can be retrieved from:
    - The predefined Hugging Face collection: "NeuralLib: Deep Learning Models for Biosignals Processing".
    - Any public or private Hugging Face repository if explicitly specified.
    """

    def __init__(self, model_name, hugging_repo=None):
        """
        Initialize the production model.

        :param model_name: (str) The name of the model to be loaded.
        :param hugging_repo: (str, optional) The full Hugging Face repository ID (e.g., "username/model_name").
            - If provided, the model will be loaded from this repository.
            - If not provided, the model will be searched in the NeuralLib collection.
            - The repository **must** contain the required files: `model_weights.pth`, `hparams.yaml`, `training_info.json`.
        """
        self.model_name = model_name
        self.local_dir = os.path.join(HUGGING_MODELS_BASE_DIR, self.model_name)

        self.hugging_repo = hugging_repo
        if self.hugging_repo is None:
            # api = HfApi()
            collection_id = "marianaagdias/neurallib-deep-learning-models-for-biosignals-processing-67473f72e30e1f0874ec5ebe"
            collection = get_collection(collection_id)
            # Find the model in the collection
            for item in collection.items:
                if item.item_id.split("/")[-1] == model_name:
                    self.hugging_repo = item.item_id
                    break

        # Ensure model files are cached locally
        self._download_and_cache_files()

        # Load model components
        self.weights_path = os.path.join(self.local_dir, "model_weights.pth")
        self.hparams_path = os.path.join(self.local_dir, "hparams.yaml")
        self.training_info_path = os.path.join(self.local_dir, "training_info.json")

        self.training_info = self._load_json(self.training_info_path)
        print(self.training_info)

        self.architecture_name = self.training_info['architecture']
        # Dynamically get the architecture class from biosignals_architectures
        self.architecture_class = getattr(arc, self.architecture_name, None)
        if not self.architecture_class:
            raise ValueError(f"Architecture {self.architecture_name} not found in biosignals_architectures.")

        super().__init__(architecture_name=self.architecture_name)  # initializing parent class

        # Initialize model state with weights
        self._initialize_model()

    def _download_and_cache_files(self):
        """Download model files from Hugging Face if not already cached locally."""
        if not os.path.exists(self.local_dir):
            try:
                print(f"Downloading model files for {self.model_name} from Hugging Face...")
                snapshot_download(repo_id=self.hugging_repo, local_dir=self.local_dir)
                print(f"Model files saved to: {self.local_dir}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model files for {self.model_name}. "
                    f"Ensure the Hugging Face repo '{self.hugging_repo}' exists and your internet connection is stable."
                    f"Error: {e}")
        else:
            print(f"Using cached model files at: {self.local_dir}")

    def _initialize_model(self):
        """Dynamically initialize the model with hyperparameters and load weights."""
        # Load hyperparameters
        self.hyperparams = self._load_yaml(self.hparams_path)

        # Dynamically initialize the architecture
        self.model = self.architecture_class(**self.hyperparams)

        # Attach metadata (training_info) to the model
        self.model.training_info = self.training_info

        # Load weights
        state_dict = torch.load(self.weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode
        print(f"✅ {self.model_name} successfully initialized.")

    def predict(self, X, gpu_id=None, post_process_fn=None, **post_process_kwargs):
        """
        Run inference on input data.
        :param X: Input tensor of shape [batch_size, seq_len, features].
        :param gpu_id: GPU ID to use for inference (if applicable).
        :param post_process_fn: Optional function for post-processing predictions.
        :return: Model predictions (post-processed if a function is provided).
        """
        # Configure device
        device = configure_device(gpu_id)
        map_location = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        print(f"Using device: {map_location}")

        # Ensure the model is on the correct device
        self.model.to(map_location)

        # Convert NumPy array to PyTorch tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor

        # Ensure input is a PyTorch tensor
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a PyTorch Tensor or NumPy array.")

        # Handle different tensor shapes
        if X.dim() == 1:
            # Case 1: 1D time series → [seq_len] → [1, seq_len, 1]
            X = X.unsqueeze(0).unsqueeze(-1)
        elif X.dim() == 2:
            # Case 2: [seq_len, features] → [1, seq_len, features]
            X = X.unsqueeze(0)
        elif X.dim() == 3:
            # Case 3: Correct shape [batch_size, seq_len, features]
            pass
        else:
            # Invalid shape
            raise ValueError(
                "Input X must have dimensions [batch_size, seq_len, features], "
                "[seq_len, features], or [seq_len]"
            )

        # Ensure input data is on the correct device
        X = X.to(map_location)

        lengths = [X.size(1)]  # Sequence length for batch size 1

        with torch.no_grad():
            output = self.model(X, lengths)  # go through the forward method of architecture

        # Apply post-processing if provided
        processed_output = post_process_fn(output, **post_process_kwargs) if post_process_fn else output

        return processed_output

    @staticmethod
    def _load_json(file_path):
        """Load JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(file_path):
        """Load YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def list_collection_models():
        """
        Lists all models in a given Hugging Face collection.
        """
        collection_id = "marianaagdias/neurallib-deep-learning-models-for-biosignals-processing-67473f72e30e1f0874ec5ebe"
        collection = get_collection(collection_id)
        for item in collection.items:
            print(item.item_id.split("/")[-1])

