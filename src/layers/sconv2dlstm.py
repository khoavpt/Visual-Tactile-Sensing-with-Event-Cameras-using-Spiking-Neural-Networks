import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

from snntorch import SpikingNeuron

class SpikingCBAM(snn.SpikingNeuron):
    """Spiking CBAM with Channel and Spatial Attention"""
    def __init__(self, 
        channels, kernel_size=3, reduction_ratio=2,
        threshold=0.5, spike_grad=snn.surrogate.fast_sigmoid(), reset_mechanism='subtract',
    ):
        
        super().__init__(threshold=threshold, spike_grad=spike_grad, reset_mechanism=reset_mechanism)
        self.channels = channels

        # Channel Attention
        self.linear1 = nn.Linear(channels, channels // reduction_ratio)
        self.linear2 = nn.Linear(channels // reduction_ratio, channels)

        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False) 

        # Register buffers to store spikes
        self.register_buffer('channel_spikes', None)
        self.register_buffer('spatial_spikes', None)

    def forward(self, x):
        """
        Args: x: (batch_size, channels, height, width)
        """
        # Channel Attention
        avg_pool = torch.mean(x, dim=[2, 3]) # (batch_size, channels)
        max_pool = torch.amax(x, dim=[2, 3]) # (batch_size, channels)

        avg_out = self.linear2(self.fire(self.linear1(avg_pool)))
        max_out = self.linear2(self.fire(self.linear1(max_pool)))

        channel_attention = self.fire(avg_out + max_out).unsqueeze(-1).unsqueeze(-1) # (batch_size, channels, 1, 1)
        x = x * channel_attention

        # Spatial Attention
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        max_pool_spatial = torch.amax(x, dim=1, keepdim=True)

        spatial_features = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)

        spatial_attention = self.fire(self.conv(spatial_features)) # (batch_size, 1, height, width)
        x = x * spatial_attention

        self.channel_spikes = channel_attention.squeeze(3)[0] # (height, width)
        self.spatial_spikes = spatial_attention[0][0] # (height, width)

        return x



class SConv2dLSTM_CBAM(SpikingNeuron):
    """
    A spiking 2d convolutional long short-term memory cell.
    Hidden states are membrane potential and synaptic current
    :math:`mem, syn`, which correspond to the hidden and cell states
    :math:`h, c` in the original LSTM formulation.

    The input is expected to be of size :math:`(N, C_{in}, H_{in}, W_{in})`
    where :math:`N` is the batch size.

    Unlike the LSTM module in PyTorch, only one time step is simulated each
    time the cell is called.

    .. math::
            \\begin{array}{ll} \\\\
            i_t = \\sigma(W_{ii} ⋆ x_t + b_{ii} + W_{hi} ⋆ mem_{t-1} + b_{hi})
            \\\\
            f_t = \\sigma(W_{if} ⋆ x_t + b_{if} + W_{hf} mem_{t-1} + b_{hf})
            \\\\
            g_t = \\tanh(W_{ig} ⋆ x_t + b_{ig} + W_{hg} ⋆ mem_{t-1} + b_{hg})
            \\\\
            o_t = \\sigma(W_{io} ⋆ x_t + b_{io} + W_{ho} ⋆ mem_{t-1} + b_{ho})
            \\\\
            syn_t = f_t ∗  c_{t-1} + i_t ∗  g_t \\\\
            mem_t = o_t ∗  \\tanh(syn_t) \\\\
        \\end{array}

    where :math:`\\sigma` is the sigmoid function, ⋆ is the 2D
    cross-correlation operator and ∗ is the Hadamard product.
    The output state :math:`mem_{t+1}` is thresholded to determine whether
    an output spike is generated.
    To conform to standard LSTM state behavior, the default reset mechanism
    is set to `reset="none"`, i.e., no reset is applied. If this is changed,
    the reset is only applied to :math:`mem_t`.

    Options to apply max-pooling or average-pooling to the state
    :math:`mem_t` are also enabled. Note that it is preferable to apply
    pooling to the state rather than the spike, as it does not make sense
    to apply pooling to activations of 1's and 0's which may lead to random
    tie-breaking.

    Padding is automatically applied to ensure consistent sizes for
    hidden states from one time step to the next.

    At the moment, stride != 1 is not supported.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn


        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                in_channels = 1
                out_channels = 8
                kernel_size = 3
                max_pool = 2
                avg_pool = 2
                flattened_input = 49 * 16
                num_outputs = 10
                beta = 0.5

                spike_grad_lstm = snn.surrogate.straight_through_estimator()
                spike_grad_fc = snn.surrogate.fast_sigmoid(slope=5)

                # initialize layers
                self.sclstm1 = snn.SConv2dLSTM(
                    in_channels,
                    out_channels,
                    kernel_size,
                    max_pool=max_pool,
                    spike_grad=spike_grad_lstm,
                )
                self.sclstm2 = snn.SConv2dLSTM(
                    out_channels,
                    out_channels,
                    kernel_size,
                    avg_pool=avg_pool,
                    spike_grad=spike_grad_lstm,
                )
                self.fc1 = nn.Linear(flattened_input, num_outputs)
                self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc)

            def forward(self, x):
                # Initialize hidden states and outputs at t=0
                syn1, mem1 = self.lif1.init_sconv2dlstm()
                syn2, mem2 = self.lif1.init_sconv2dlstm()
                mem3 = self.lif3.init_leaky()

                # Record the final layer
                spk3_rec = []
                mem3_rec = []

                # Number of steps assuming x is [N, T, C, H, W] with
                # N = Batches, T = Time steps, C = Channels,
                # H = Height, W = Width
                num_steps = x.size()[1]

                for step in range(num_steps):
                    x_step = x[:, step, :, :, :]
                    spk1, syn1, mem1 = self.sclstm1(x_step, syn1, mem1)
                    spk2, syn2, mem2 = self.sclstm2(spk1, syn2, mem2)
                    cur = self.fc1(spk2.flatten(1))
                    spk3, mem3 = self.lif1(cur, mem3)

                    spk3_rec.append(spk3)
                    mem3_rec.append(mem3)

                return torch.stack(spk3_rec), torch.stack(mem3_rec)


    :param in_channels: number of input channels
    :type in_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int, tuple, or list

    :param bias: If `True`, adds a learnable bias to the output. Defaults to
        `True`
    :type bias: bool, optional

    :param max_pool: Applies max-pooling to the hidden state :math:`mem`
        prior to thresholding if specified. Defaults to 0
    :type max_pool: int, tuple, or list, optional

    :param avg_pool: Applies average-pooling to the hidden state :math:`mem`
        prior to thresholding if specified. Defaults to 0
    :type avg_pool: int, tuple, or list, optional

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        ATan surrogate gradient
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to \
    :math:`mem` each time the threshold is met. Reset-by-subtraction: \
        "subtract", reset-to-zero: "zero, none: "none". Defaults to "none"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and \
    :math:`syn` are quantized to a valid state for the forward pass. \
        Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, syn_0, mem_0
        - **input_** of shape `(batch, in_channels, H, W)`: tensor \
        containing input features
        - **syn_0** of shape `(batch, out_channels, H, W)`: tensor \
        containing the initial synaptic current (or cell state) for each \
        element in the batch.
        - **mem_0** of shape `(batch, out_channels, H, W)`: tensor \
        containing the initial membrane potential (or hidden state) for each \
        element in the batch.

    Outputs: spk, syn_1, mem_1
        - **spk** of shape `(batch, out_channels, H/pool, W/pool)`: tensor \
        containing the output spike (avg_pool and max_pool scale if greater \
        than 0.)
        - **syn_1** of shape `(batch, out_channels, H, W)`: tensor \
        containing the next synaptic current (or cell state) for each element \
        in the batch
        - **mem_1** of shape `(batch, out_channels, H, W)`: tensor \
        containing the next membrane potential (or hidden state) for each \
        element in the batch

    Learnable Parameters:
        - **SConv2dLSTM.conv.weight** (torch.Tensor) - the learnable \
        weights, of shape ((in_channels + out_channels), 4*out_channels, \
        kernel_size).

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        max_pool=0,
        avg_pool=0,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="none",
        state_quant=False,
        output=False,

        cbam_kernel_size=3,
        cbam_reduction_ratio=2,
    ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self.avg_pool = avg_pool
        self.bias = bias
        self._sconv2dlstm_cases()

        # padding is essential to keep same shape for next step
        if type(self.kernel_size) is int:
            self.padding = kernel_size // 2, kernel_size // 2
        else:
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.out_channels,
            out_channels=4 * self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.cbam = SpikingCBAM(self.out_channels, cbam_kernel_size, cbam_reduction_ratio)
        
        # Register buffers to store spikes
        self.register_buffer('spk_list', None)

    def _init_mem(self):
        syn = torch.zeros(0)
        mem = torch.zeros(0)

        self.register_buffer("syn", syn, False)
        self.register_buffer("mem", mem, False)



    def reset_mem(self):
        self.syn = torch.zeros_like(self.syn, device=self.syn.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.syn, self.mem





    def init_sconv2dlstm(self):
        """Deprecated, use :class:`SConv2dLSTM.reset_mem` instead"""
        return self.reset_mem()





    def forward(self, input_, syn=None, mem=None):
        if not syn == None:
            self.syn = syn

        if not mem == None:
            self.mem = mem

        if self.init_hidden and (not mem == None or not syn == None):
            raise TypeError(
                "`mem` or `syn` should not be passed as an argument while `init_hidden=True`"
            )

        size = input_.size()
        correct_shape = (size[0], self.out_channels, size[2], size[3])
        if not self.syn.shape == correct_shape:
            self.syn = torch.zeros(correct_shape, device=self.syn.device)

        if not self.mem.shape == correct_shape:
            self.mem = torch.zeros(correct_shape, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)
        self.syn, self.mem = self.state_function(input_)

        if self.state_quant:
            self.syn = self.state_quant(self.syn)
            self.mem = self.state_quant(self.mem)

        if self.max_pool:
            self.spk = self.fire(F.max_pool2d(self.mem, self.max_pool))
        elif self.avg_pool:
            self.spk = self.fire(F.avg_pool2d(self.mem, self.avg_pool))
        else:
            self.spk = self.fire(self.mem)

        self.spk_list = self.spk[0][0] # (height, width)

        if self.output:
            return self.spk, self.syn, self.mem
        elif self.init_hidden:
            return self.spk
        else:
            return self.spk, self.syn, self.mem

    def _base_state_function(self, input_):
        combined = torch.cat(
            [input_, self.mem], dim=1
        )  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.out_channels, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        f = self.cbam(f)

        base_fn_syn = f * self.syn + i * g
        base_fn_mem = o * torch.tanh(base_fn_syn)

        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        combined = torch.cat(
            [input_, self.mem], dim=1
        )  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.out_channels, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        base_fn_syn = f * self.syn + i * g
        base_fn_mem = o * torch.tanh(base_fn_syn)

        return 0, base_fn_mem

    def _base_sub(self, input_):
        syn, mem = self._base_state_function(input_)
        mem -= self.reset * self.threshold
        return syn, mem

    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = self._base_state_reset_zero(input_)
        syn2 *= self.reset
        mem2 *= self.reset
        syn -= syn2
        mem -= mem2
        return syn, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def _sconv2dlstm_cases(self):
        if self.max_pool and self.avg_pool:
            raise ValueError(
                "Only one of either `max_pool` or `avg_pool` may be "
                "specified, not both."
            )



    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SConv2dLSTM_CBAM):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()





    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are
        instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SConv2dLSTM_CBAM):
                cls.instances[layer].syn = torch.zeros_like(
                    cls.instances[layer].syn,
                    device=cls.instances[layer].syn.device,
                )
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
