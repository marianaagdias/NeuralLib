import json
import datetime
import os
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import configure_seed, configure_device, DatasetSequence, collate_fn
from utils.plots import LossPlotCallback


class BaseModel(pl.LightningModule):
    def __init__(self, model_name, training_info=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.training_info = training_info if training_info else {}

    def structure(self):
        """Returns the structure of the module (architecture, number of layers, nodes)."""
        return str(self)

    def checkpoints(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}"
        dir_name = f"{self.model_name}_{arch_string}_dt{timestamp}"
        checkpoints_directory = os.path.join(self.results_directory, 'checkpoints', dir_name)
        os.makedirs(checkpoints_directory, exist_ok=True)
        return checkpoints_directory

    def save_training_information(self, trainer, optimizer, train_dataset_name, val_loss, total_training_time, retraining):
        """Save information about the training process."""
        self.training_info = {
            'train_dataset': train_dataset_name,
            'epochs': trainer.current_epoch,
            'optimizer': str(optimizer),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'validation_loss': val_loss,
            'training_time': total_training_time,  # If tracked manually
            'retraining': retraining
        }

    def save_training_information_to_file(self, directory):
        """Saves trained information to a JSON file."""
        training_info = self.training_info
        training_info_file = os.path.join(directory, 'training_info.json')

        if not os.path.exists(training_info_file):
            with open(training_info_file, 'w') as f:
                json.dump(training_info, f, indent=4)

    def train_model(self, path_x, path_y, patience, epochs, all_samples=False, samples=None, dataset_name=None,
                    pretrained_checkpoint=None, classification=False):
        # if pretrained_checkpoint is provided (path to a .ckpt file with the trained weights of the model), these
        # weights are imported and the training is done from there

        if dataset_name is None:
            raise ValueError("You must provide the 'dataset_name' for tracking the model's training process.")

        # Configure seed and device
        configure_seed(42)
        device = configure_device(self.gpu_id)

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

        # Initialize the model
        model = self  # same as doing: model = GRUseq2seq(n_features=self.n_features, hid_dim=self.hid_dim,...)

        # if it is to re-train a model, we need to load the trained weights
        if pretrained_checkpoint is not None:
            model = self.load_from_checkpoint(
                pretrained_checkpoint,  # directory
                n_features=self.n_features,
                hid_dim=self.hid_dim,
                n_layers=self.n_layers,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                results_directory=self.results_directory,
                gpu_id=self.gpu_id,
                bidirectional=self.bidirectional,
                criterion=self.criterion,
            )
            print(f"Loaded model from checkpoint: {pretrained_checkpoint}")

        # Define the model checkpoint callback
        arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}"
        checkpoints_directory = model.checkpoints()
        retrained_suffix = "_retrained" if pretrained_checkpoint else ""
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoints_directory,
            filename=f"{self.model_name}_{arch_string}{retrained_suffix}",
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
        plot_callback = LossPlotCallback(save_path=os.path.join(checkpoints_directory, "loss_plot.png"))

        start_time = time.time()

        # PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if device != 'cpu' else 'cpu',
            devices=[device] if device != 'cpu' else 1,
            default_root_dir=checkpoints_directory,
            callbacks=[checkpoint_callback, early_stopping_callback, plot_callback]
        )

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloader)

        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")

        # Best validation loss
        val_loss = checkpoint_callback.best_model_score.item()  # it's working!

        # optimizer
        optimizer = model.configure_optimizers()

        # Save the training information in the model
        model.save_training_information(
            trainer=trainer,
            optimizer=optimizer,
            train_dataset_name=dataset_name,
            val_loss=val_loss,
            total_training_time=total_training_time,
            retraining=True if pretrained_checkpoint else False
        )

        # print and save the model training information
        print(model.training_info)
        model.save_training_information_to_file(checkpoints_directory)

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"best_model_path: {best_checkpoint_path}")




