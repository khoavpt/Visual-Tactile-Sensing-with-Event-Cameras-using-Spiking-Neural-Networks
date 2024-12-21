import pytorch_lightning as pl
import matplotlib.pyplot as plt

class LossLogger(pl.Callback):
    def __init__(self, model_name):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_error_counts = [] # list of (sequence index, error_count_of_each_sequence)
        self.model_name = model_name
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        train_accuracy = trainer.callback_metrics.get('train_accuracy')
        self.train_losses.append(train_loss.item())
        self.train_accuracies.append(train_accuracy.item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_accuracy = trainer.callback_metrics.get('val_accuracy')
        self.val_losses.append(val_loss.item())
        self.val_accuracies.append(val_accuracy.item())

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.legend()
        plt.suptitle(f'Training and Validation Loss and Accuracy of the {self.model_name} model')
        plt.show()
