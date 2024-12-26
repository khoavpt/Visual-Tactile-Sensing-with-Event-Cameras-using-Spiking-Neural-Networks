import pytorch_lightning as pl
import matplotlib.pyplot as plt
import time

class LossLogger(pl.Callback):
    def __init__(self, model_name):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_f1s = []
        self.val_f1s = []
        self.epoch_durations = []
        self.model_name = model_name
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_durations.append(epoch_duration)
        
        train_loss = trainer.callback_metrics.get('train_loss')
        train_accuracy = trainer.callback_metrics.get('train_accuracy')
        train_f1 = trainer.callback_metrics.get('train_f1')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_accuracy is not None:
            self.train_accuracies.append(train_accuracy.item())
        if train_f1 is not None:
            self.train_f1s.append(train_f1.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_accuracy = trainer.callback_metrics.get('val_accuracy')
        val_f1 = trainer.callback_metrics.get('val_f1')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy.item())
        if val_f1 is not None:
            self.val_f1s.append(val_f1.item())

    def plot_results(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.legend()
        plt.suptitle(f'Training and Validation Loss, Accuracy of the {self.model_name} model')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()