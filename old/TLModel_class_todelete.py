import torch
import torch.nn as nn
import json
import os
import time
from base import Model
from modules import Module
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import importlib.util
from torch.utils.data import DataLoader
from utils import configure_seed, configure_device, DatasetSequence, collate_fn, LossPlotCallback


class TLModel(Model):
    def __init__(self, model_name, modules_list=None, layers=None):
        """
        Initialize a TLModel using pretrained modules and potentially new layers.

        :param model_name: Name of the transfer-learning model.
        :param modules_list: List of Module objects to be used in the TLModel.
        :param layers: List of nn.Modules to be added to the model for customization.
        """
        super(TLModel, self).__init__(model_name=model_name)

        # List to store pretrained modules
        self.modules_list = modules_list if modules_list else []

        # List to store additional layers
        self.layers = nn.ModuleList(layers) if layers else nn.ModuleList()

        # Combine pretrained modules and new layers
        self.full_model = nn.Sequential(*[mod.architecture for mod in self.modules_list], *self.layers)

        # To store training details if the model has been trained
        self.training_info = {}

    def add_pretrained_module(self, module):
        """
        Add a pretrained module to the TLModel
        :param module: instance of Module
        """
        if not isinstance(module, Module):
            raise ValueError("Only objects of type 'Module' can be added.")
        self.modules_list.append(module)
        self.full_model.add_module(module.module_name, module.architecture)

    def add_layers(self, layers):
        """
        Add additional layers (torch.nn.Modules) to the TLModel.
        :param layers: A list of nn.Module instances to add on top of the existing architecture.
        """
        if not all(isinstance(layer, nn.Module) for layer in layers):
            raise ValueError("All layers must be instances of nn.Module")
        for layer in layers:
            self.layers.append(layer)
            self.full_model.add_module(f"layer_{len(self.layers)}", layer)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the TLModel. Uses the full architectue (modules + layers).
        """
        return self.full_model(*args, **kwargs)

    # TODO: É NECESSARIO ESTE MÉTODO???
    def save_model(self, save_directory):
        """
        Save the entire model (architecture, weights, training_info) to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        weights_save_path = os.path.join(save_directory, f"{self.model_name}_weights.pth")
        torch.save(self.full_model.state_dict(), weights_save_path)

        training_info_path = os.path.join(save_directory, f"{self.model_name}_training_info.json")
        with open(training_info_path, 'w') as f:
            json.dump(self.training_info, f, indent=4)

        print(f"TLModel {self.model_name} saved with weights at {weights_save_path}.")

    def freeze_modules(self):
        """
        Freeze all pretrained modules, allowing only new layers to be trained.
        """

        for mod in self.modules_list:
            mod.freeze()
        print(f"All pretrained modules in {self.module_name} have been frozen.")

    def unfreeze_modules(self):
        """
        Unfreeze all pretrained modules to allow them to be retrained.
        """
        for mod in self.modules_list:
            mod.unfreeze()
        print(f"All pretrained modules in {self.module_name} have been unfrozen.")

    # TODO: ESTÁ COPY DO BASEMODEL --- É PRECISO MODIFICAR PARA O CASO DE ESTARMOS A RE-TREINAR O MODELO
    def retrain_model(self, path_x, path_y, patience, batch_size, epochs, all_samples=False, samples=None, dataset_name=None,
                    trained_for=None, classification=False, enable_tensorboard=False):

        # Configure seed and device
        configure_seed(42)
        device = configure_device(self.gpu_id)

        # Check for TensorBoard availability
        tensorboard_available = importlib.util.find_spec("tensorboard") is not None
        if enable_tensorboard and not tensorboard_available:
            print("Warning: TensorBoard is not installed. Proceeding without TensorBoard logging.")
            enable_tensorboard = False

        if dataset_name is None:
            raise ValueError("You must provide the 'dataset_name' for tracking the model's training process.")
        if trained_for is None:
            raise ValueError("You must provide the 'trained_for' input with the task that this model is intended to "
                             "perform, for tracking the model's training process.")

        # Initialize the model
        model = self  # same as doing: model = GRUseq2seq(n_features=self.n_features, hid_dim=self.hid_dim,...)
        if self.checkpoints_directory is None:
            raise ValueError("TLmodel")
        else:
            self.create_checkpoints_directory()

        # Datasets and Dataloaders
        train_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='train', all_samples=all_samples,
                                        samples=samples)
        val_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='val', all_samples=all_samples,
                                      samples=samples)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Calculate class weights based on the training dataset -- only for classification problems
        # if classification:
        #     class_weights = calculate_class_weights(train_dataset)
        # Convert class_weights to the GPU
        # class_weights = class_weights.to(device=device)

        # Define the model callbacks
        arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}"
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.checkpoints_directory,
            filename=f"{self.model_name}_{arch_string}",
            save_top_k=1,
            mode='min'
        )

        # Define early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )

        # Loss plotting callback
        plot_callback = LossPlotCallback(save_path=os.path.join(self.checkpoints_directory, "loss_plot.png"))

        # Initialize Trainer with TensorBoardLogger if enabled
        if enable_tensorboard and tensorboard_available:
            logger = TensorBoardLogger(self.checkpoints_directory, name="tensorboard_logs")
            print(f"TensorBoard logs will be saved to {logger.log_dir}")
        else:
            logger = None  # No logging if TensorBoard is not available or enabled

        # Start training with PyTorch Lightning Trainer
        start_time = time.time()
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if device != 'cpu' else 'cpu',
            devices=[device] if device != 'cpu' else 1,
            default_root_dir=self.checkpoints_directory,
            callbacks=[checkpoint_callback, early_stopping_callback, plot_callback],
            logger=logger  # TensorBoard logger added if available
        )

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloader)

        # Save training information
        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")
        val_loss = checkpoint_callback.best_model_score.item()  # Best validation loss
        optimizer = model.configure_optimizers()[0][0]

        model.save_training_information(
            trainer=trainer,
            optimizer=optimizer,
            train_dataset_name=dataset_name,
            trained_for=trained_for,
            val_loss=val_loss,
            total_training_time=total_training_time
        )
        print(model.training_info)
        model.save_training_information_to_file(self.checkpoints_directory)

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Training complete. Best_model_path: {best_checkpoint_path}")

