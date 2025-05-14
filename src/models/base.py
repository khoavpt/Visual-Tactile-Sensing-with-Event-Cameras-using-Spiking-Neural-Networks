import abc
import snntorch as snn
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional as F
from ..utils.visualization import log_spike_activity_to_wandb

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
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4], device=self.device))
        for t in range(sequence.size(1)):
            loss += loss_fn(output[:, t], target[:, t])
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
        
        metrics = {
            'loss/train': loss,
            'accuracy/train': accuracy,
            'f1/train': f1,
            'precision/train': precision,
            'recall/train': recall
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1, precision, recall = self.common_step(batch, batch_idx)
        
        metrics = {
            'loss/val': loss,
            'accuracy/val': accuracy, 
            'f1/val': f1,
            'precision/val': precision,
            'recall/val': recall
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss/val',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def test_step(self, batch, batch_idx):
        sequence, target = batch  
        batch_size, sequence_length = sequence.shape[0], sequence.shape[1]
        
        hidden_states = self.init_hidden_states()
        outputs = []
        
        for t in range(sequence_length):
            x = sequence[:, t]
            output, hidden_states = self.process_frame(x, hidden_states)
            outputs.append(output)
            
        outputs = torch.stack(outputs, dim=1)  # (batch_size, sequence_length, num_classes)
        
        # Calculate loss
        loss = 0
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4], device=self.device))
        for t in range(sequence_length):
            loss += loss_fn(outputs[:, t], target[:, t])
        loss /= sequence_length
        
        # Calculate accuracy
        preds = outputs.argmax(dim=-1)  # (batch_size, sequence_length)
        correct = (preds == target).sum()
        total = target.numel()
        accuracy = correct / total
        
        # Calculate F1 score
        f1 = F.f1_score(preds.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')
        # Calculate precision and recall
        precision = F.precision(preds.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')
        recall = F.recall(preds.view(-1), target.view(-1), num_classes=2, average='macro', task='binary')
        

        # Log metrics
        metrics = {
            'loss/test': loss,
            'accuracy/test': accuracy,
            'f1/test': f1,
            'precision/test': precision,
            'recall/test': recall

        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log spike activity
        if batch_idx == 2:  # Only log for first batch to avoid too much data
            log_spike_activity_to_wandb(self, batch)
        
        return loss

    
    def on_test_epoch_end(self):
        # Fetch logged test metrics from trainer
        avg_metrics = {
            'test_accuracy': self.trainer.callback_metrics['accuracy/test'],
            'test_f1': self.trainer.callback_metrics['f1/test'],
            'test_precision': self.trainer.callback_metrics['precision/test'],
            'test_recall': self.trainer.callback_metrics['recall/test']
        }
        
        # Log final test metrics
        self.log_dict(avg_metrics)

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
