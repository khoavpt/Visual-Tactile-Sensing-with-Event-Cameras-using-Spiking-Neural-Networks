import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNN(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class ConvLSTM(pl.LightningModule):
    def __init__(self, in_channels, feature_size):
        super().__init__()
        self.cnn = CNN(in_channels, feature_size)
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
        sequence, target = batch # Sequence: (batch_size, sequence_length, channels, height, width), target: (batch_size, sequence_length)
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
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        return optimizer
