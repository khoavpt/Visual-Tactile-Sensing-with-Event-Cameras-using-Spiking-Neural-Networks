import snntorch as snn
import torch
import torch.nn as nn

from .base import BaseSpikingModel

class ConvSNN(BaseSpikingModel):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", in_channels=1, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.lif1 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch_size, 6, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.lif2 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch_size, 16, 5, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.lif3 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, learn_beta=True, learn_threshold=True, threshold=0.5)
        self.fc2 = nn.Linear(120, 2)
        self.lif4 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, learn_beta=True, learn_threshold=True, threshold=0.1)

    def init_hidden_states(self):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        return mem1, mem2, mem3, mem4

    def process_frame(self, x, hidden_states):
        """ 
        Process a single frame
        Args:
            x: (batch_size, channels, height, width)
            hidden_states: Tuple of hidden states to carry over timesteps (mem1, mem2, mem3, mem4)
        """
        batch_size, channels, height, width = x.size()
        mem1, mem2, mem3, mem4 = hidden_states

        x = self.conv1(x)
        spk1, mem1 = self.lif1(x, mem1)
        x = self.pool1(spk1)

        x = self.conv2(x)
        spk2, mem2 = self.lif2(x, mem2)
        x = self.pool2(spk2)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        spk3, mem3 = self.lif3(x, mem3)

        x = self.fc2(spk3)
        spk4, mem4 = self.lif4(x, mem4)

        return mem4, (mem1, mem2, mem3, mem4)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            output: (batch_size, sequence_length, num_classes)
        """
        batch_size, sequence_length, channels, height, width = x.size()
        hidden_states = self.init_hidden_states()

        outputs = []
        for t in range(sequence_length):
            frame = x[:, t]  # Extract frame at time t
            output, hidden_states = self.process_frame(frame, hidden_states)
            outputs.append(output)

        return torch.stack(outputs, dim=1)  # (batch_size, sequence_length, num_classes)