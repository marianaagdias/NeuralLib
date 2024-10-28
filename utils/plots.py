import matplotlib.pyplot as plt
import pytorch_lightning as pl


class LossPlotCallback(pl.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Log the training loss for this epoch
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log the validation loss for this epoch
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())

    def on_train_end(self, trainer, pl_module):
        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path)
        plt.close()
