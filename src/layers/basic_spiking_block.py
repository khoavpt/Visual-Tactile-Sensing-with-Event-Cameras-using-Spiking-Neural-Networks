import torch
import torch.nn as nn
import snntorch as snn
# import spconv.pytorch as spconv

from .batchnorm import tdBatchNorm2d, tdBatchNorm1d

class ConvSpikingBlock(nn.Module):
    """
    Convolutional Spiking Block: Conv2d -> tdBN2d -> LIF -> Pool
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 alpha=1,  VTH=1.0, 
                 spike_grad=snn.surrogate.fast_sigmoid(), beta_init=0.9, learn_beta=True, learn_threshold=True,
                 pooling_layer=None):
        """
        Args:
            - Conv args:
                in_channels: Number of input channels
                out_channels: Number of output channels
                kernel_size: Size of the convolving kernel
                stride: Stride of the convolution
                padding: Zero-padding added to both sides of the input
            - tdBN args:
                alpha: Additional parameter which may change in resblock
                VTH: Threshold voltage
            - LIF args:
                spike_grad: Surrogate gradient for spiking neurons
                beta_init: Initial value of beta
                learn_beta: Whether to learn beta
                learn_threshold: Whether to learn threshold
            - pooling_layer: Pooling layer to apply after LIF
        """
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = tdBatchNorm2d(out_channels, alpha=alpha, VTH=VTH)
        self.lif = snn.Leaky(beta=beta_init, spike_grad=spike_grad, threshold=VTH, 
                            learn_beta=learn_beta, learn_threshold=learn_threshold)
        self.pooling_layer = pooling_layer if pooling_layer else nn.Identity()
        
        # Register buffer to store spikes
        self.register_buffer('spk_list', None)

    def forward(self, x, mem_init):
        """x: (batch_size, seq_len, channels, height, width)"""
        batch_size, sequence_length, channels, height, width = x.size()

        x = x.view(-1, channels, height, width)
        x = self.conv(x)
        x = x.view(batch_size, sequence_length, *x.shape[1:]) # (batch_size, seq_len, channels, height, width)
        x = self.bn(x)

        spk_list = []
        mem = mem_init
        for t in range(sequence_length):
            spk, mem = self.lif(x[:, t], mem)
            spk = self.pooling_layer(spk)
            spk_list.append(spk)

        return torch.stack(spk_list, dim=1)

    def fuse_weight(self):
        """
        Batchnorm-scale fusion for inference (CURRENTLY NOT FUSING WEIGHTS!!!!!!!)
        """
        with torch.no_grad():
            fused_weight, fused_bias = self.bn.fuse_weight(self.conv.weight, self.conv.bias)
            self.conv.weight.data.copy_(fused_weight)
            self.conv.bias.data.copy_(fused_bias)

    def process_frame(self, x, mem_prev):
        """
        Inference: Skip batch norm, use pre-fused weights (CURRENTLY NOT FUSING WEIGHTS!!!!!!!)

        x: (batch_size, channels, height, width)
        """
        x = self.conv(x)
        # x = self.bn(x.unsqueeze(1)).squeeze(1) # (batch_size, channels, height, width)
        spk, mem = self.lif(x, mem_prev) # (batch_size, channels, height, width)
        x = self.pooling_layer(spk)
        self.spk_list = spk[0][0]
        return x, mem

class LinearSpikingBlock(nn.Module):
    """
    Linear Spiking Block: Linear -> tdBN1d -> LIF
    """
    def __init__(self, in_features, out_features, 
                 alpha=1, VTH=1.0, 
                 spike_grad=snn.surrogate.fast_sigmoid(), beta_init=0.9, learn_beta=True, learn_threshold=True):
        """
        Args:
            - Linear args:
                in_features: Size of each input sample
                out_features: Size of each output sample
            - tdBN args:
                alpha: Additional parameter which may change in resblock
                VTH: Threshold voltage
            - LIF args:
                spike_grad: Surrogate gradient for spiking neurons
                beta_init: Initial value of beta
                learn_beta: Whether to learn beta
                learn_threshold: Whether to learn threshold
        """
        super().__init__()
        
        self.fc = nn.Linear(in_features, out_features)
        self.bn = tdBatchNorm1d(out_features, alpha=alpha, VTH=VTH)
        self.lif = snn.Leaky(beta=beta_init, spike_grad=spike_grad, threshold=VTH, 
                            learn_beta=learn_beta, learn_threshold=learn_threshold)
                
        # Register buffer to store spikes
        self.register_buffer('spk_list', None)

    def forward(self, x, mem_init):
        """x: (batch_size, seq_len, in_features)"""
        batch_size, sequence_length, in_features = x.size()

        x = x.view(-1, in_features)

        x = self.fc(x)
        x = x.view(batch_size, sequence_length, -1) # (batch_size, seq_len, out_features)
        x = self.bn(x)

        spk_list = []
        mem = mem_init
        for t in range(sequence_length):
            spk, mem = self.lif(x[:, t], mem) # (batch_size, out_features)
            spk_list.append(spk)

        return torch.stack(spk_list, dim=1)
    
    def fuse_weight(self):
        """
        Batchnorm-scale fusion for inference (CURRENTLY NOT FUSING WEIGHTS!!!!!!!)
        """
        fused_weight, fused_bias = self.bn.fuse_weight(self.fc.weight, self.fc.bias)
        self.fc.weight.data.copy_(fused_weight)
        self.fc.bias.data.copy_(fused_bias)

    def process_frame(self, x, mem_prev):
        """
        Inference: Skip batch norm, use pre-fused weights (CURRENTLY NOT FUSING WEIGHTS!!!!!!!)

        x: (batch_size, in_features)
        """
        x = self.fc(x)
        # x = self.bn(x.unsqueeze(1)).squeeze(1) # Apply BN using running stats in eval mode
        spk, mem = self.lif(x, mem_prev) # spk: (batch_size, out_features)
        # Store spikes for visualization
        self.spk_list = spk[0].unsqueeze(1) # (out_features, 1)
        return spk, mem
    
# class SparseConvSpikingBlock(nn.Module):
#     """
#     Sparse Convolutional Spiking Block: SparseConv -> tdBN2d -> LIF -> Pool
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
#                  alpha=1, VTH=1.0,
#                  spike_grad=snn.surrogate.fast_sigmoid(), beta_init=0.9, learn_beta=True, learn_threshold=True,
#                  pooling_layer=None):
#         super().__init__()
        
#         # Replace standard Conv2d with SparseConv
#         self.conv = spconv.SparseConv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=True
#         )
        
#         self.bn = tdBatchNorm2d(out_channels, alpha=alpha, VTH=VTH)
#         self.lif = snn.Leaky(beta=beta_init, spike_grad=spike_grad, threshold=VTH,
#                             learn_beta=learn_beta, learn_threshold=learn_threshold)
#         self.pooling_layer = pooling_layer if pooling_layer else nn.Identity()
        
#         # Register buffer to store spikes
#         self.register_buffer('spk_list', None)

#     def forward(self, x, mem_init):
#         """
#         x: (batch_size, seq_len, channels, height, width) containing sparse tensors
#         """
#         batch_size, sequence_length, channels, height, width = x.size()
        
#         # Convert dense tensor to sparse format
#         spk_list = []
#         mem = mem_init
        
#         for t in range(sequence_length):
#             # Get current timestep
#             x_t = x[:, t]  # (batch_size, channels, height, width)
            
#             # Convert to sparse format
#             x_t = x_t.to_sparse()
            
#             # Apply sparse convolution
#             x_t = self.conv(x_t)
            
#             # Convert back to dense for batch norm and LIF
#             x_t = x_t.to_dense()
            
#             # Apply batch norm
#             x_t = self.bn(x_t)
            
#             # Apply LIF
#             spk, mem = self.lif(x_t, mem)
            
#             # Apply pooling if specified
#             spk = self.pooling_layer(spk)
            
#             spk_list.append(spk)
            
#         return torch.stack(spk_list, dim=1)

#     def process_frame(self, x, mem_prev):
#         """
#         Inference: Skip batch norm, use pre-fused weights
#         x: (batch_size, channels, height, width) sparse tensor
#         """
#         # Convert to sparse tensor
#         indices = torch.nonzero(x)
#         features = x[tuple(indices.t())]
        
#         sparse_x = spconv.SparseConvTensor(
#             features,
#             indices,
#             spatial_shape=x.shape[2:],
#             batch_size=x.shape[0]
#         )
        
#         # Apply sparse convolution
#         x = self.conv(sparse_x).dense()
#         spk, mem = self.lif(x, mem_prev)
#         x = self.pooling_layer(spk)
#         # Store spikes for visualization
#         self.spk_list = spk.unsqueeze(1)  # Add time dimension
#         return x, mem
    
#     def fuse_weight(self):
#         """
#         Batchnorm-scale fusion for inference
#         """
#         with torch.no_grad():
#             # Get the original weight and bias
#             original_weight = self.conv.weight.dense()  # Convert sparse weight to dense
#             original_bias = self.conv.bias

#             # Fuse with batch norm
#             fused_weight, fused_bias = self.bn.fuse_weight(original_weight, original_bias)
            
#             # Convert fused weight back to sparse format
#             self.conv.weight = spconv.SparseConvTensor.from_dense(fused_weight)
#             self.conv.bias.data.copy_(fused_bias)