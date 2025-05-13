import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional as F
from pl_bolts.modules.conv_lstm import ConvLSTM

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=8,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=12,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=12,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return x

class ConvLSTM(pl.LightningModule):
    def __init__(self, in_channels, feature_size, lr=0.001):
        super().__init__()
        self.lr = lr
        self.cnn = CNN(in_channels)
        self.convlstm = ConvLSTM(input_size=(8, 8),  # Assuming the output size of CNN is (16, 16)
                            input_dim=16,  # Number of input channels from CNN
                            hidden_dim=[32],  # Number of output channels for LSTM
                            kernel_size=(3, 3),
                            num_layers=1,
                            batch_first=True,
                            bias=True,
                            return_all_layers=False)

        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        """
        batch_size, sequence_length, channels, height, width = x.size()
        x = x.view(batch_size * sequence_length, channels, height, width)
        x = self.cnn(x) #   (batch_size * sequence_length, output_size)
        x = x.view(batch_size, sequence_length, -1) # (batch_size, sequence_length, output
        x, _ = self.lstm(x) # (batch_size, sequence_length, hidden_size)
        x = self.fc(x) # (batch_size, sequence_length, 2)
        return x

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

        return loss, accuracy, f1

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)

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