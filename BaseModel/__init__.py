import json
import datetime
import os
import glob
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import importlib.util
from torch.utils.data import DataLoader
from utils import configure_seed, configure_device, DatasetSequence, collate_fn
from utils.plots import LossPlotCallback


class BaseModel(pl.LightningModule):
    def __init__(self, model_name, training_info=None, checkpoints_directory=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.training_info = training_info if training_info else {}
        self.checkpoints_directory = checkpoints_directory

    def structure(self):
        """Returns the structure of the module (architecture, number of layers, nodes)."""
        return str(self)

    def create_checkpoints(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}"
        dir_name = f"{self.model_name}_{arch_string}_dt{timestamp}"
        checkpoints_directory = os.path.join(self.results_directory, 'checkpoints', dir_name)
        os.makedirs(checkpoints_directory, exist_ok=True)
        self.checkpoints_directory = checkpoints_directory
        return checkpoints_directory

    def save_training_information(self, trainer, optimizer, train_dataset_name, trained_for, val_loss, total_training_time):
        """Save information about the training process."""
        self.training_info = {
            'train_dataset': train_dataset_name,
            'task': trained_for,
            'epochs': trainer.current_epoch,
            'optimizer': str(optimizer),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'validation_loss': val_loss,
            'training_time': total_training_time,  # If tracked manually
            # 'retraining': retraining
        }

    def save_training_information_to_file(self, directory):
        """Saves trained information to a JSON file."""
        training_info = self.training_info
        training_info_file = os.path.join(directory, 'training_info.json')

        if not os.path.exists(training_info_file):
            with open(training_info_file, 'w') as f:
                json.dump(training_info, f, indent=4)

    def train_model(self, path_x, path_y, patience, epochs, all_samples=False, samples=None, dataset_name=None,
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
        if self.checkpoints_directory is not None:
            raise ValueError("The model has been trained. If it is to be retrained, TLModel.retrain should be used.")
        else:
            self.create_checkpoints()

        # Datasets and Dataloaders
        train_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='train', all_samples=all_samples,
                                        samples=samples)
        val_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='val', all_samples=all_samples,
                                      samples=samples)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

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
            total_training_time=total_training_time,
        )
        print(model.training_info)
        model.save_training_information_to_file(self.checkpoints_directory)

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Training complete. Best_model_path: {best_checkpoint_path}")




