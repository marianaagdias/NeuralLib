import architectures as arc
import torch
import json
import os
import yaml
# from transformers import AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import HfApi, create_repo, snapshot_download, list_models


class ProductionModel(arc.Architecture):
    """
    Production Models
    A class for trained models, extending Architecture by adding weights and training information.
    Includes methods for uploading to and importing from Hugging Face.
    """

    def __init__(self, model_name, weights_path=None, training_info_path=None, yaml_path=None, checkpoints_directory=None,
                 hugging_repo=None):
        super(ProductionModel, self).__init__(model_name=model_name)

        # Determine the source of model data
        if checkpoints_directory:
            self.weights_path, self.training_info = arc.get_weights_and_info_from_checkpoints(checkpoints_directory)
            self.archi_params = arc.get_hparams_from_checkpoints(checkpoints_directory)
        elif hugging_repo:
            self.weights_path, self.training_info = arc.get_weights_and_info_from_hugging(hugging_repo)
            self.archi_params = arc.get_hparams_from_hugging(hugging_repo)
        elif weights_path and training_info_path and yaml_path:
            self.weights_path = weights_path
            self.archi_params = self._load_yaml(yaml_path)
            self.training_info = self._load_json(training_info_path)
        else:
            raise ValueError("In order to call the model, it is necessary that the weights, architecture parameters and"
                             "training information are provided (the paths to each file). These can be provided "
                             "separately or by providing the checkpoints_directory or the hugging face repository.")

        # Initialize model state with weights
        self._initialize_state()

    def _initialize_state(self):
        """Load weights into the model."""
        try:
            state_dict = torch.load(self.weights_path, map_location='cpu')
            self.load_state_dict(state_dict)
            print(f"Loaded weights from {self.weights_path}")
        except Exception as e:
            raise ValueError(f"Failed to load weights from {self.weights_path}. Error: {e}")

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

    def upload_to_huggingface(self, repo_name, token):
        """
        Uploads the model's weights and training info to Hugging Face's Model Hub.

        :param repo_name: Name of the Hugging Face repository.
        :param token: Hugging Face token for authentication.
        """
        collection_name = "NeuralLib: Deep Learning Models for Biosignals Processing"

        # Create a Hugging Face repo
        api = HfApi()
        repo_url = create_repo(name=repo_name, token=token, exist_ok=True)

        # Save weights locally for upload
        local_dir = f"./{repo_name}"  # TODO: CHANGE to proper dir of the model
        os.makedirs(local_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(local_dir, "pytorch_model.bin"))

        # TODO: THIS PART (TRAINING INFORMATION) should be to get the model's training info and not save it
        #  (it is saved during/after training)
        # Save training information as JSON
        training_info_file = os.path.join(local_dir, "training_info.json")
        with open(training_info_file, "w") as f:
            json.dump(self.training_info, f, indent=4)

        # Upload files to Hugging Face
        api.upload_folder(repo_id=repo_name,
                          folder_path=local_dir,
                          token=token)
        print(f"Model uploaded to Hugging Face repository: {repo_url}")
        # Add model to collection
        api.update_repo_visibility(repo_id=repo_name, token=token, collection=collection_name)
        print(f"Model added to collection: {collection_name}")

    @staticmethod
    def list_collection_models(collection_name, token):
        """
        Lists all models in a given Hugging Face collection.

        :param collection_name: Name of the collection.
        :param token: Hugging Face token for authentication.
        :return: List of models in the collection.
        """
        models = list_models(search=collection_name, token=token)
        return [model.modelId for model in models]



