import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional as F

class ConvSNN_L(pl.LightningModule):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", in_channels=1, L=10, lr=0.001):
        super().__init__()
        self.lr = lr
        self.L = L

        if spikegrad == 'fast_sigmoid':
            spikegrad = snn.surrogate.fast_sigmoid()
        elif spikegrad == 'arctan':
            spikegrad = snn.surrogate.atan()
        elif spikegrad == 'heaviside':
            spikegrad = snn.surrogate.heaviside()
        else:
            raise ValueError(f"Unknown surrogate gradient function: {spikegrad}")

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta_init, spike_grad=spikegrad, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch_size, 6, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta_init, spike_grad=spikegrad, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch_size, 16, 5, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.lif3 = snn.Leaky(beta=beta_init, spike_grad=spikegrad, learn_beta=True, learn_threshold=True, threshold=0.5)
        self.fc2 = nn.Linear(120, 2)
        self.lif4 = snn.Leaky(beta=beta_init, spike_grad=spikegrad, learn_beta=True, learn_threshold=True, threshold=0.1)

        self.fc3 = nn.Linear(L * 2, 2)

    def forward(self, x):
        """
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)
        """
        batch_size, sequence_length, channels, height, width = x.size()
        

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []
        mem4_rec = []
        out_rec = []

        for t in range(sequence_length):
            x_t = x[:, t] # (batch_size, channels, height, width)
            x_t = self.conv1(x_t) # (batch_size, 6, 28, 28)
            spk1, mem1 = self.lif1(x_t, mem1) # (batch_size, 6, 28, 28)
            x_t = self.pool1(spk1) # (batch_size, 6, 14, 14)

            x_t = self.conv2(x_t) # (batch_size, 16, 10, 10)
            spk2, mem2 = self.lif2(x_t, mem2) # (batch_size, 16, 10, 10)
            x_t = self.pool2(spk2) # (batch_size, 16, 5, 5)

            x_t = x_t.view(batch_size, -1) # (batch_size, 16 * 5 * 5)
            x_t = self.fc1(x_t) # (batch_size, 120)
            spk3, mem3 = self.lif3(x_t, mem3) # (batch_size, 120)
            x_t = self.fc2(spk3) # (batch_size, output_size)
            spk4, mem4 = self.lif4(x_t, mem4) # (batch_size, output_size)

            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

            if t < self.L - 1:
                spk4_rec_L = torch.stack(spk4_rec[0:t + 1], dim=1) # (batch_size, t + 1, output_size)
                zeros_tensor = torch.zeros((batch_size, self.L - t - 1, 2)).to('cuda')
                spk4_rec_L = torch.cat((zeros_tensor, spk4_rec_L), dim=1).reshape(batch_size, -1) # (batch_size, L * output_size)
            else:
                spk4_rec_L = torch.stack(spk4_rec[t-self.L+1:t+1], dim=1).reshape(batch_size, -1) # (batch_size, L * output_size)

            out = self.fc3(spk4_rec_L) # (batch_size, output_size)
            out_rec.append(out)
        return torch.stack(out_rec, dim=1)# (batch_size, sequence_length,  output_size)

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