import torch
import torch.nn as nn


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
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, VTH=1.0):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.VTH = VTH

    def forward(self, x):
        """
        x: (batch_size, time_steps, channels, height, width)
        """
        exponential_average_factor = 0.0
    
        if self.training:
            if self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            mean = x.mean([0, 1, 3, 4]) 
            var = x.var([0, 1, 3, 4], unbiased=False)
            n = x.numel() / x.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean 
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var

            x_hat = self.alpha * self.VTH * (x - mean[None, None, :, None, None]) / (torch.sqrt(var[None, None, :, None, None] + self.eps))
            if self.affine:
                x_hat = x_hat * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        else:
            x_hat = self.alpha * self.VTH * (x - self.running_mean[None, None, :, None, None]) / (torch.sqrt(self.running_var[None, None, :, None, None] + self.eps))
            if self.affine:
                x_hat = x_hat * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        return x_hat
    
    def fuse_weight(self, weight, bias):
        """BatchNorm-scale fusion for inference"""
        if self.running_var is None or self.running_mean is None:
            raise ValueError("tdBN must be trained before fusing weights.")

        scale = (self.weight * self.alpha * self.VTH) / torch.sqrt(self.running_var + self.eps)
        fused_weight = weight * scale.view(-1, 1, 1, 1)
        fused_bias = scale * (bias - self.running_mean) + self.bias

        return fused_weight, fused_bias
    
class tdBatchNorm1d(nn.BatchNorm1d):
    """Implementation of Time-dependent BatchNorm1d"""
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, VTH=1.0):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.VTH = VTH

    def forward(self, x):
        """
        x: (batch_size, time_steps, features)
        """
        exponential_average_factor = 0.0

        if self.training:
            if self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        exponential_average_factor = self.momentum
                
            mean = x.mean(dim=(0, 1))
            var = x.var(dim=(0, 1), unbiased=False)
            n = x.numel() / x.size(2)

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
            
            x_hat = self.alpha * self.VTH * (x - mean[None, None, :]) / torch.sqrt(var[None, None, :] + self.eps)
            if self.affine:
                x_hat = x_hat * self.weight[None, None, :] + self.bias[None, None, :]

        else:
            x_hat = self.alpha * self.VTH * (x - self.running_mean[None, None, :]) / torch.sqrt(self.running_var[None, None, :] + self.eps)
            if self.affine:
                x_hat = x_hat * self.weight[None, None, :] + self.bias[None, None, :]

        return x_hat
    
    def fuse_weight(self, weight, bias):
        """Fuses BatchNorm into weights for inference."""
        if self.running_var is None or self.running_mean is None:
            raise ValueError("tdBatchNorm1d must be trained before fusing weights.")

        scale = (self.weight * self.alpha * self.VTH) / torch.sqrt(self.running_var + self.eps)
        fused_weight = weight * scale.view(-1, 1)
        
        fused_bias = scale * (bias - self.running_mean) + self.bias

        return fused_weight, fused_bias