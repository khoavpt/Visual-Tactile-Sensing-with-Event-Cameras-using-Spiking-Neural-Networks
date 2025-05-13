import torch.nn as nn

from .base import BaseSpikingModel
from ..layers.basic_spiking_block import ConvSpikingBlock, LinearSpikingBlock

class ConvSNN(BaseSpikingModel):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", in_channels=1, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()

        # Conv block 1
        self.conv_block1 = ConvSpikingBlock(
            in_channels=in_channels, out_channels=6, kernel_size=5, 
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
            pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Conv block 2
        self.conv_block2 = ConvSpikingBlock(
            in_channels=6, out_channels=16, kernel_size=5, 
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
            pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Linear block 1
        self.linear_block1 = LinearSpikingBlock(
            in_features=16 * 5 * 5, out_features=120,
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
        )

        # Linear block 2
        self.linear_block2 = LinearSpikingBlock(
            in_features=120, out_features=2,
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
        )

    
    def init_hidden_states(self):
        mem1 = self.conv_block1.lif.init_leaky()
        mem2 = self.conv_block2.lif.init_leaky()
        mem3 = self.linear_block1.lif.init_leaky()
        mem4 = self.linear_block2.lif.init_leaky()
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

        x, mem1 = self.conv_block1.process_frame(x, mem1)
        x, mem2 = self.conv_block2.process_frame(x, mem2) # x: ()
        x = x.view(batch_size, -1)  # Flatten
        x, mem3 = self.linear_block1.process_frame(x, mem3)
        x, mem4 = self.linear_block2.process_frame(x, mem4)

        return x, (mem1, mem2, mem3, mem4)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            output: (batch_size, sequence_length, num_classes)
        """
        batch_size, sequence_length, channels, height, width = x.size()
        mem1, mem2, mem3, mem4 = self.init_hidden_states()

        x = self.conv_block1(x, mem1)
        x = self.conv_block2(x, mem2)
        x = x.view(batch_size, sequence_length, -1)  # Flatten
        x = self.linear_block1(x, mem3)
        x = self.linear_block2(x, mem4)

        return x  # Output shape: (batch_size, sequence_length, num_classes)
    
    def to_inference_mode(self):
        """
        Switch model to inference mode (to eval + fuse bn-scale)
        """
        self.eval()
        self.conv_block1.fuse_weight()
        self.conv_block2.fuse_weight()
        self.linear_block1.fuse_weight()
        self.linear_block2.fuse_weight()