import torch
import torch.nn as nn

from .base import BaseSpikingModel
from ..layers.basic_spiking_block import ConvSpikingBlock, LinearSpikingBlock
from ..layers.sconv2dlstm import SConv2dLSTM_CBAM

class SpikingConvLSTM_CBAM(BaseSpikingModel):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", in_channels=1, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()

        # Conv block 1 
        self.conv_block1 = ConvSpikingBlock(
            in_channels=in_channels, out_channels=8, kernel_size=5, padding=2,
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
            # pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv block 2
        self.conv_block2 = ConvSpikingBlock(
            in_channels=8, out_channels=12, kernel_size=3, padding=1,
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
            pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv block 3
        self.conv_block3 = ConvSpikingBlock(
            in_channels=12, out_channels=16, kernel_size=3, padding=1,
            alpha=1, VTH=0.5,
            spike_grad=self.spikegrad, beta_init=0.9,
            pooling_layer=nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # SConv2dLSTM
        self.sconv2dlstm = SConv2dLSTM_CBAM(
            in_channels=16, out_channels=12, kernel_size=3, max_pool=2, threshold=0.5,
            cbam_kernel_size=3, cbam_reduction_ratio=2,
        )

        # Linear block 1
        self.linear_block1 = LinearSpikingBlock(
            in_features=12*4*4, out_features=2,
            alpha=1, VTH=0.3,
            spike_grad=self.spikegrad, beta_init=0.9,
        )
        

    def init_hidden_states(self):
        mem1 = self.conv_block1.lif.init_leaky()
        mem2 = self.conv_block2.lif.init_leaky()
        mem3 = self.conv_block3.lif.init_leaky()
        syn4, mem4 = self.sconv2dlstm.reset_mem()
        mem5 = self.linear_block1.lif.init_leaky()
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

        x, mem1 = self.conv_block1.process_frame(x, mem1) # (batch_size, 8, 32, 32)
        x, mem2 = self.conv_block2.process_frame(x, mem2) # (batch_size, 12, 16, 16)
        x, mem3 = self.conv_block3.process_frame(x, mem3) # (batch_size, 16, 8, 8)
 
        x, syn4, mem4 = self.sconv2dlstm(x, syn4, mem4) # (batch_size, 12, 4, 4)
        x = x.view(batch_size, -1) # (batch_size, 12*4*4)
        x, mem5 = self.linear_block1.process_frame(x, mem5) # x: (batch_size, 2)
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

        x = self.conv_block1(x, mem1) # (batch_size, sequence_length, 6, 32, 32)
        x = self.conv_block2(x, mem2) # (batch_size, sequence_length, 12, 16, 16)
        x = self.conv_block3(x, mem3) # (batch_size, sequence_length, 16, 8, 8)

        outputs = []
        for t in range(sequence_length):
            # spk4, syn4, mem4 = self.slstm(x[:, t], syn4, mem4)
            # out = self.linear_block2(spk4.unsqueeze(1), mem5)
            # outputs.append(out[:, 0])  # 
            spk4, syn4, mem4 = self.sconv2dlstm(x[:, t], syn4, mem4) # spk3: (batch_size, 12, 4, 4)
            out = spk4.view(batch_size, 1, -1) # (batch_size, 1, 6*5*5)
            out = self.linear_block1(out, mem5) # x: (batch_size, 1, 2)
            outputs.append(out[:, 0]) # (batch_size, 2)

        return torch.stack(outputs, dim=1)
    
    def to_inference_mode(self):
        """
        Switch model to inference mode (eval + fuse bn-scale)
        """
        self.eval()
        # self.conv_block1.fuse_weight()
        # self.conv_block2.fuse_weight()
        # self.conv_block3.fuse_weight()
        # self.linear_block1.fuse_weight()
    
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