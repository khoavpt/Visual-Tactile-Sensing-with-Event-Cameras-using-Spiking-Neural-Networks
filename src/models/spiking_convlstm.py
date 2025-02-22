import snntorch as snn
import torch
import torch.nn as nn

from .base import BaseSpikingModel

from src.models.convlstm import ConvLSTM

class SpikingConvLSTM(BaseSpikingModel):
    def __init__(self, beta_init, feature_size=64, spikegrad="fast_sigmoid", in_channels=1, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.lif1 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, reset_mechanism="subtract", threshold=0.5, learn_beta=True, learn_threshold=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch_size, 6, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.lif2 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, reset_mechanism="subtract", threshold=0.5, learn_beta=True, learn_threshold=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch_size, 16, 5, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, feature_size)
        self.lif3 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, reset_mechanism="subtract", threshold=0.5, learn_beta=True, learn_threshold=True)

        self.slstm = snn.SLSTM(input_size=feature_size,
                               hidden_size=128,
                               spike_grad=self.spikegrad,
                               threshold=0.5,
                               reset_mechanism="subtract")
        
        self.fc2 = nn.Linear(128, 2)
        self.lif4 = snn.Leaky(beta=self.beta_init, spike_grad=self.spikegrad, reset_mechanism="subtract", learn_beta=True, learn_threshold=True, threshold=0.5)

    def init_hidden_states(self):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        syn4, mem4 = self.slstm.init_slstm()
        mem5 = self.lif4.init_leaky()
        return mem1, mem2, mem3, syn4, mem4, mem5

    def process_frame(self, x, hidden_states):
        """ 
        Process a single frame
        Args:
            x: (batch_size, channels, height, width)
            hidden_states: Tuple of hidden states to carry over timesteps (mem1, mem2, mem3, syn4, mem4, mem5)
        """
        batch_size, channels, height, width = x.size()
        mem1, mem2, mem3, syn4, mem4, mem5 = hidden_states

        x = self.conv1(x)
        spk1, mem1 = self.lif1(x, mem1)
        x = self.pool1(spk1)

        x = self.conv2(x)
        spk2, mem2 = self.lif2(x, mem2)
        x = self.pool2(spk2)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        spk3, mem3 = self.lif3(x, mem3)

        spk4, syn4, mem4 = self.slstm(spk3, syn4, mem4)
        x = self.fc2(spk4)
        spk5, mem5 = self.lif4(x, mem5)

        return mem5, (mem1, mem2, mem3, syn4, mem4, mem5)

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