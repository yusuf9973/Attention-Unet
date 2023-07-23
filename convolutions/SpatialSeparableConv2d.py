import torch
import torch.nn as nn

class SpatialSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super(SpatialSeparableConv2d, self).__init__()
        assert isinstance(kernel_size, int) # only support square kernels
        self.conv1 = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), padding=(padding, 0), groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding), bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
