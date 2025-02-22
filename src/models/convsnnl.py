import snntorch as snn
import torch
import torch.nn as nn

from .base import BaseSpikingModel

class ConvSNN_L(BaseSpikingModel):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", in_channels=1, L=10, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()
        self.L = L

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

        self.fc3 = nn.Linear(L * 2, 2)


    def init_hidden_states(self):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_buffer = []      
        return spk4_buffer, mem1, mem2, mem3, mem4
    
    def process_frame(self, x, hidden_states):
        """ 
        Process a single frame
        Args:
            x: (batch_size, channels, height, width)
            hidden_states: Tuple of hidden states to carry over timesteps (mem1, mem2, mem3, mem4)
        """
        batch_size, channels, height, width = x.size()
        spk4_buffer, mem1, mem2, mem3, mem4 = hidden_states

        if  len(spk4_buffer) == 0:
            spk4_buffer = [torch.zeros(batch_size, 2).to(x.device) for _ in range(self.L)]

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

        # Update spike history buffer
        spk4_buffer.append(mem4) # Currently using membrane potential instead of spikes
        spk4_buffer.pop(0)

        spk4_rec_L = torch.stack(spk4_buffer, dim=1)
        spk4_rec_L = spk4_rec_L.reshape(batch_size, -1) # (batch_size, L * 2)
        out = self.fc3(spk4_rec_L)

        return out, (spk4_buffer, mem1, mem2, mem3, mem4)
    
    # def forward(self, x):
    #     """
    #         Args:
    #             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)
    #     """
    #     batch_size, sequence_length, channels, height, width = x.size()

    #     hidden_state = self.init_hidden_states()

    #     spk4_rec = []
    #     mem4_rec = []
    #     out_rec = []

    #     for t in range(sequence_length):
    #         x_t = x[:, t] # (batch_size, channels, height, width)
    #         x_t = self.conv1(x_t) # (batch_size, 6, 28, 28)
    #         spk1, mem1 = self.lif1(x_t, mem1) # (batch_size, 6, 28, 28)
    #         x_t = self.pool1(spk1) # (batch_size, 6, 14, 14)

    #         x_t = self.conv2(x_t) # (batch_size, 16, 10, 10)
    #         spk2, mem2 = self.lif2(x_t, mem2) # (batch_size, 16, 10, 10)
    #         x_t = self.pool2(spk2) # (batch_size, 16, 5, 5)

    #         x_t = x_t.view(batch_size, -1) # (batch_size, 16 * 5 * 5)
    #         x_t = self.fc1(x_t) # (batch_size, 120)
    #         spk3, mem3 = self.lif3(x_t, mem3) # (batch_size, 120)
    #         x_t = self.fc2(spk3) # (batch_size, output_size)
    #         spk4, mem4 = self.lif4(x_t, mem4) # (batch_size, output_size)

    #         spk4_rec.append(spk4)
    #         mem4_rec.append(mem4)

    #         if t < self.L - 1:
    #             spk4_rec_L = torch.stack(spk4_rec[0:t + 1], dim=1) # (batch_size, t + 1, output_size)
    #             zeros_tensor = torch.zeros((batch_size, self.L - t - 1, 2)).to('cuda')
    #             spk4_rec_L = torch.cat((zeros_tensor, spk4_rec_L), dim=1).reshape(batch_size, -1) # (batch_size, L * output_size)
    #         else:
    #             spk4_rec_L = torch.stack(spk4_rec[t-self.L+1:t+1], dim=1).reshape(batch_size, -1) # (batch_size, L * output_size)

    #         out = self.fc3(spk4_rec_L) # (batch_size, output_size)
    #         out_rec.append(out)
    #     return torch.stack(out_rec, dim=1)# (batch_size, sequence_length,  output_size)

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