import abc
import snntorch as snn
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional as F

class BaseSpikingModel(pl.LightningModule):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", lr=0.001):
        super().__init__()
        self.lr = lr
        self.beta_init = beta_init

        if spikegrad == 'fast_sigmoid':
            self.spikegrad = snn.surrogate.fast_sigmoid()
        elif spikegrad == 'arctan':
            self.spikegrad = snn.surrogate.atan()
        elif spikegrad == 'heaviside':
            self.spikegrad = snn.surrogate.heaviside()
        else:
            raise ValueError(f"Unknown surrogate gradient function: {spikegrad}")

    def common_step(self, batch, batch_idx):
        sequence, target = batch  # Sequence: (batch_size, sequence_length, channels, height, width), target: (batch_size, sequence_length)
        output = self(sequence)  # (batch_size, sequence_length, num_classes)

        # Calculate loss
        loss = 0
        for t in range(sequence.size(1)):
            loss += nn.CrossEntropyLoss()(output[:, t], target[:, t])
        loss /= sequence.size(1)  # Average loss over the sequence length

        # Calculate accuracy
        preds = output.argmax(dim=-1)  # (batch_size, sequence_length)
        correct = (preds == target).sum()
        total = target.numel()
        accuracy = correct / total

        # Calculate F1 score
        f1 = F.f1_score(preds.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')

        # Calculate precision and recall
        precision = F.precision(preds.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')
        recall = F.recall(preds.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')

        return loss, accuracy, f1, precision, recall

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1, precision, recall = self.common_step(batch, batch_idx)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_precision', precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recall', recall, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1, precision, recall = self.common_step(batch, batch_idx)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        sequence, target = batch  # Sequence: (batch_size, sequence_length, channels, height, width), target: (batch_size, sequence_length)
        hidden_states = self.init_hidden_states()

        predictions = []
        for t in range(sequence.size(1)):
            output, hidden_states = self.process_frame(sequence[:, t], hidden_states)
            predictions.append(output.argmax(dim=1))  # (batch_size, num_classes)

        predictions = torch.stack(predictions, dim=1)  # (batch_size, sequence_length, num_classes)

        # Calculate accuracy
        correct = (predictions == target).sum()
        total = target.numel()
        accuracy = correct / total

        # Calculate f1 score
        f1 = F.f1_score(predictions.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')
        precision = F.precision(predictions.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')
        recall = F.recall(predictions.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')

        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', recall, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    @abc.abstractmethod
    def init_hidden_states(self) -> tuple:
        raise NotImplementedError
    
    @abc.abstractmethod
    def process_frame(self, x, hidden_states) -> tuple:
        """ 
        Process a single frame (for inference)
        Args:
            x: (batch_size, channels, height, width)
            hidden_states: Tuple of hidden states to carry over timesteps
        Returns:
            output: (batch_size, num_classes)
            hidden_states: Tuple of hidden states to carry over timesteps
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_inference_mode(self):
        """
        Switch model to inference mode (to eval + fuse bn-scale)
        """
        raise NotImplementedError