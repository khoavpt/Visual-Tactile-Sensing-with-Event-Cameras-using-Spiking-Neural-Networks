import snntorch as snn
import torch
import torch.nn as nn

from .base import BaseSpikingModel
from ..layers.basic_spiking_block import ConvSpikingBlock, LinearSpikingBlock

class SpikingConvLSTM(BaseSpikingModel):
    def __init__(self, beta_init, feature_size=64, spikegrad="fast_sigmoid", in_channels=1, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()

        # Conv block 1
        self.conv_block1 = ConvSpikingBlock(
            in_channels=in_channels, out_channels=6, kernel_size=5, 
            alpha=1, VTH=0.8,
            spike_grad=self.spikegrad, beta_init=0.9,
            pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Conv block 2
        self.conv_block2 = ConvSpikingBlock(
            in_channels=6, out_channels=16, kernel_size=5, 
            alpha=1, VTH=0.8,
            spike_grad=self.spikegrad, beta_init=0.9,
            pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Linear block 1
        self.linear_block1 = LinearSpikingBlock(
            in_features=16 * 5 * 5, out_features=feature_size,
            alpha=1, VTH=0.8,
            spike_grad=self.spikegrad, beta_init=0.9,
        )
        # SLSTM
        self.slstm = snn.SLSTM(
            input_size=feature_size, hidden_size=128, spike_grad=self.spikegrad, threshold=0.5, reset_mechanism="subtract"
        )
        # Linear block 2
        self.linear_block2 = LinearSpikingBlock(
            in_features=128, out_features=2,
            alpha=1, VTH=0.8,
            spike_grad=self.spikegrad, beta_init=0.9,
        )

    def init_hidden_states(self):
        mem1 = self.conv_block1.lif.init_leaky()
        mem2 = self.conv_block2.lif.init_leaky()
        mem3 = self.linear_block1.lif.init_leaky()
        syn4, mem4 = self.slstm.init_slstm()
        mem5 = self.linear_block2.lif.init_leaky()
        return mem1, mem2, mem3, syn4, mem4, mem5
    
    def process_frame(self, x, hidden_states):
        """
        Process a single frame
        Args:
            x: (batch_size, channels, height, width)
            hidden_states: Tuple of hidden states
        """
        batch_size = x.size(0)
        mem1, mem2, mem3, syn4, mem4, mem5 = hidden_states

        x, mem1 = self.conv_block1.process_frame(x, mem1)
        x, mem2 = self.conv_block2.process_frame(x, mem2)
        x = x.view(batch_size, -1)  
        x, mem3 = self.linear_block1.process_frame(x, mem3)
        spk4, syn4, mem4 = self.slstm(x, syn4, mem4)
        x, mem5 = self.linear_block2.process_frame(spk4, mem5)

        return x, (mem1, mem2, mem3, syn4, mem4, mem5)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            output: (batch_size, sequence_length, num_classes)
        """
        batch_size, sequence_length, channels, height, width = x.size()
        mem1, mem2, mem3, syn4, mem4, mem5 = self.init_hidden_states()

        x = self.conv_block1(x, mem1)
        x = self.conv_block2(x, mem2)
        x = x.view(batch_size, sequence_length, -1) 
        x = self.linear_block1(x, mem3)
        
        outputs = []
        for t in range(sequence_length):
            spk4, syn4, mem4 = self.slstm(x[:, t], syn4, mem4)
            out = self.linear_block2(spk4.unsqueeze(1), mem5)
            outputs.append(out[:, 0])  # 

        return torch.stack(outputs, dim=1)
    
    def to_inference_mode(self):
        """
        Switch model to inference mode (eval + fuse bn-scale)
        """
        self.eval()
        self.conv_block1.fuse_weight()
        self.conv_block2.fuse_weight()
        self.linear_block1.fuse_weight()
        self.linear_block2.fuse_weight()
    
    # def _init_parameters(self):
    #     """
    #     ANN to SNN weight initialization
    #     """
    #     print("Loading pretrained weight from ConvLSTM model ...")
    #     checkpoint_path = os.path.join(ROOTPATH, 'saved_models/convlstm_accumulator/epoch=49-step=350.ckpt')
    #     model = ConvLSTM.load_from_checkpoint(checkpoint_path, in_channels=1, feature_size=64)

    #     self.conv1.weight = model.cnn.conv_block_1[0].weight
    #     self.conv1.bias = model.cnn.conv_block_1[0].bias

    #     self.conv2.weight = model.cnn.conv_block_2[0].weight
    #     self.conv2.bias = model.cnn.conv_block_2[0].bias

    #     self.fc1.weight = model.cnn.fc1.weight
    #     self.fc1.bias = model.cnn.fc1.bias

    #     self.fc2.weight = model.fc.weight
    #     self.fc2.bias = model.fc.bias

    #     self.slstm.lstm_cell._parameters['weight_ih'] = model.lstm._parameters['weight_ih_l0']
    #     self.slstm.lstm_cell._parameters['weight_hh'] = model.lstm._parameters['weight_hh_l0']

    #     self.slstm.lstm_cell._parameters['bias_ih'] = model.lstm._parameters['bias_ih_l0']
    #     self.slstm.lstm_cell._parameters['bias_hh'] = model.lstm._parameters['bias_hh_l0']
    #     print("Pretrained weight loaded successfully.")