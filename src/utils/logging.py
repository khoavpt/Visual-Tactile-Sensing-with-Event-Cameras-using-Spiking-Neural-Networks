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
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.epoch_durations = []
        self.model_name = model_name
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_durations.append(epoch_duration)
        
        train_loss = trainer.callback_metrics.get('loss/train')
        train_accuracy = trainer.callback_metrics.get('accuracy/train')
        train_f1 = trainer.callback_metrics.get('f1/train')
        train_precision = trainer.callback_metrics.get('precision/train')
        train_recall = trainer.callback_metrics.get('recall/train')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_accuracy is not None:
            self.train_accuracies.append(train_accuracy.item())
        if train_f1 is not None:
            self.train_f1s.append(train_f1.item())
        if train_precision is not None:
            self.train_precisions.append(train_precision.item())
        if train_recall is not None:
            self.train_recalls.append(train_recall.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('loss/val')
        val_accuracy = trainer.callback_metrics.get('accuracy/val')
        val_f1 = trainer.callback_metrics.get('f1/val')
        val_precision = trainer.callback_metrics.get('precision/val')
        val_recall = trainer.callback_metrics.get('recall/val')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy.item())
        if val_f1 is not None:
            self.val_f1s.append(val_f1.item())
        if val_precision is not None:
            self.val_precisions.append(val_precision.item())
        if val_recall is not None:
            self.val_recalls.append(val_recall.item())

    def plot_results(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training and Validation Loss')

        # Accuracy
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title('Training and Validation Accuracy')

        # Precision
        axes[1, 0].plot(self.train_precisions, label='Train Precision')
        axes[1, 0].plot(self.val_precisions, label='Validation Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].set_title('Training and Validation Precision')

        # Recall
        axes[1, 1].plot(self.train_recalls, label='Train Recall')
        axes[1, 1].plot(self.val_recalls, label='Validation Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].set_title('Training and Validation Recall')

        plt.suptitle(f'Training and Validation Metrics of the {self.model_name} model')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
