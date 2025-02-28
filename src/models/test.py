import torch.nn as nn
import torch
import snntorch as snn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.functional as F

from .base import BaseSpikingModel

class tdLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to convert.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """
    def __init__(self, layer, bn=None, steps=300):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.steps = steps

    def forward(self, x):
        print(f"tdLayer input shape: {x.shape}")
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (self.steps,), device=x.device)
        for step in range(self.steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        print(f"tdLayer output shape: {x_.shape}")
        return x_

class tdBatchNorm2d(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, vth, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = vth

    def forward(self, input):
        """
        Args:
            input: (batch, seq_len, C, H, W)
        Returns:
            output: (batch, seq_len, C, H, W)
        """
        # print(f"tdBatchNorm2d input shape: {input.shape}")
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.vth * (input - mean[None, None, :, None, None]) / (torch.sqrt(var[None, None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        # print(f"tdBatchNorm2d output shape: {input.shape}")
        return input

class tdBatchNorm1d(nn.BatchNorm1d):
    """
    tdBN1d: Batch Normalization with time-domain averaging for 1D inputs.
    Related paper: https://arxiv.org/pdf/2011.05280
    
    Args:
        num_features (int): Number of features in the input.
        vth (float): Threshold voltage.
        eps (float): A small value to avoid division by zero.
        momentum (float): Momentum for running statistics.
        alpha (float): Scaling factor.
        affine (bool): Whether to use learnable affine parameters.
        track_running_stats (bool): Whether to track running statistics.
    """
    def __init__(self, num_features, vth, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = vth

    def forward(self, input):

        # print(f"tdBatchNorm1d input shape: {input.shape}")
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1])  # Averaging over batch and time steps
            var = input.var([0, 1], unbiased=False)  # Compute variance over batch and time steps
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.vth * (input - mean[None, None, :]) / (torch.sqrt(var[None, None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :] + self.bias[None, None, :]

        # print(f"tdBatchNorm1d output shape: {input.shape}")
        return input
    
class ConvSNNBlock(nn.Module):
    def __init__(self, in_features, out_features, beta, spikegrad, block_type="conv", kernel_size=5, pool=True, threshold=0.5, alpha=1):
        super(ConvSNNBlock, self).__init__()
        self.block_type = block_type  # "conv" hoặc "fc"
        self.in_features = in_features
        self.out_features = out_features

        if block_type == "conv":
            self.layer = nn.Conv2d(in_features, out_features, kernel_size=kernel_size)
            self.tdbn = tdBatchNorm2d(out_features, threshold, alpha=alpha)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None
        elif block_type == "fc":
            self.layer = nn.Linear(in_features, out_features)
            self.tdbn = tdBatchNorm1d(out_features, threshold, alpha=alpha)
            self.pool = None  # Không có Pooling cho FC

        self.lif = snn.Leaky(beta=beta, spike_grad=spikegrad, threshold=threshold, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        # print(f"\nConvSNNBlock ({self.block_type}: {self.in_features}->{self.out_features}) input shape: {x.shape}")
        batch_size, seq_len = x.shape[:2]

        # Vòng for 1: Conv hoặc FC
        if self.block_type == "conv":
            conv_outputs = []
            for t in range(seq_len):
                out = self.layer(x[:, t])
                conv_outputs.append(out)
                # if t == 0:  # Print only for the first time step
                    # print(f"  Conv layer output shape (t=0): {out.shape}")
            out = torch.stack(conv_outputs, dim=1)  # (batch, seq_len, C, H, W)
        else:  # FC
            fc_outputs = []
            for t in range(seq_len):
                flattened = x[:, t].view(batch_size, -1)
                # if t == 0:  # Print only for the first time step
                    # print(f"  Flattened input shape (t=0): {flattened.shape}")
                out = self.layer(flattened)
                # if t == 0:  # Print only for the first time step
                    # print(f"  FC layer output shape (t=0): {out.shape}")
                fc_outputs.append(out)
            out = torch.stack(fc_outputs, dim=1)  # (batch, seq_len, features)

        # print(f"  After layer stack shape: {out.shape}")

        # Normalize với tdbn
        out = self.tdbn(out)

        # Vòng for 2: LIF
        lif_out = []
        mem = self.lif.init_leaky()
        for t in range(seq_len):
            spk, mem = self.lif(out[:, t], mem)
            lif_out.append(spk)
            # if t == 0:  # Print only for the first time step
                # print(f"  LIF output shape (t=0): {spk.shape}")

        lif_out = torch.stack(lif_out, dim=1)
        # print(f"  After LIF stack shape: {lif_out.shape}")

        # Pooling if needed (only for Conv blocks)
        if self.pool and self.block_type == "conv":
            pooled_outputs = []
            for t in range(seq_len):
                pooled = self.pool(lif_out[:, t])
                pooled_outputs.append(pooled)
                # if t == 0:  # Print only for the first time step
                    # print(f"  Pool output shape (t=0): {pooled.shape}")
            lif_out = torch.stack(pooled_outputs, dim=1)
            # print(f"  After pooling stack shape: {lif_out.shape}")

        # print(f"ConvSNNBlock output shape: {lif_out.shape}")
        return lif_out


class ConvSNN(BaseSpikingModel):
    def __init__(self, beta_init, spikegrad="fast_sigmoid", in_channels=1, lr=0.001):
        super().__init__(beta_init=beta_init, spikegrad=spikegrad, lr=lr)
        self.save_hyperparameters()

        self.block1 = ConvSNNBlock(in_channels, 6, beta=self.beta_init, spikegrad=self.spikegrad, block_type="conv")
        self.block2 = ConvSNNBlock(6, 16, beta=self.beta_init, spikegrad=self.spikegrad, block_type="conv")  
        self.block3 = ConvSNNBlock(16 * 5 * 5, 128, beta=self.beta_init, spikegrad=self.spikegrad, block_type="fc")  # Fully Connected 1
        self.block4 = ConvSNNBlock(128, 2, beta=self.beta_init, spikegrad=self.spikegrad, threshold=0.1, alpha=1/(2**0.5), block_type="fc")  # Fully Connected 2 (output)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            output: (batch_size, sequence_length, num_classes)
        """
        # print(f"\n--- Starting forward pass ---")
        # print(f"Input shape: {x.shape}")
        
        x = self.block1(x)
        # print(f"\nAfter block1 shape: {x.shape}")
        
        x = self.block2(x)
        # print(f"\nAfter block2 shape: {x.shape}")
        
        x = self.block3(x)
        # print(f"\nAfter block3 shape: {x.shape}")
        
        x = self.block4(x)
        # print(f"\nFinal output shape: {x.shape}")

        return x  # (batch_size, sequence_length, num_classes)

# # Test code to show the shapes
# if __name__ == "__main__":
#     # Example input tensor: (batch_size=2, sequence_length=10, channels=1, height=28, width=28)
#     # This resembles MNIST-like data across 10 time steps
#     batch_size = 8
#     seq_len = 300
#     input_tensor = torch.randn(batch_size, seq_len, 1, 32, 32)
    
#     # Create model
#     model = ConvSNN(beta_init=0.9, spikegrad="fast_sigmoid", in_channels=1)
    
#     # Forward pass
#     print("\n=== STARTING MODEL TEST ===")
#     output = model(input_tensor)
#     print("\n=== MODEL TEST COMPLETE ===")
#     print(f"Final output shape: {output.shape}")