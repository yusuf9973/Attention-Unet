import torch
import torch.nn as nn
def conv2d(in_channels, out_channels, kernel_size, padding, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
