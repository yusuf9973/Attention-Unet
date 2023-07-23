import torch
import torch.nn as nn
import torch.nn.functional as F
from convolutions.gaussian_dynamic_conv import gaussian_dynamic_conv
from convolutions.deformable_conv import deformable_conv
from convolutions.Adaptive_deformable_conv import Adaptive_deformable_conv

class orgACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(orgACBlock, self).__init__()
        c1 = int(out_channels*0.33)
        c2 = int(out_channels*0.33)
        c3 = out_channels - c1 - c2
        self.square_conv = nn.Conv2d(in_channels,c1 , (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.ver_conv = nn.Conv2d(in_channels,c2 , (kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)
        self.hor_conv = nn.Conv2d(in_channels, c3, (1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        x1 = self.square_conv(x)
        x2 = self.ver_conv(x)
        x3 = self.hor_conv(x)
        x = torch.cat([x1,x2,x3], dim=1) 
        return x

class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ACBlock, self).__init__()
        c1 = int(out_channels*0.5)
        c2 = int(out_channels*0.33)
        c3 = out_channels - c1 - c2
        self.square_conv = nn.Conv2d(in_channels,c1 , (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.ver_conv = nn.Conv2d(in_channels,c2 , (kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)
        self.hor_conv = nn.Conv2d(in_channels, c3, (1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        x1 = self.square_conv(x)
        x2 = self.ver_conv(x)
        x3 = self.hor_conv(x)
        x = torch.cat([x1,x2,x3], dim=1)
        return x



class GD_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(GD_AC, self).__init__()
        assert isinstance(kernel_size, int)
        self.square_conv = gaussian_dynamic_conv(in_channels, int(out_channels*0.5), kernel_size, padding, bias=False)
        self.ver_conv = nn.Conv2d(in_channels, int(out_channels*0.33), (kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)
        self.hor_conv = nn.Conv2d(in_channels, int(out_channels*0.167), (1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x1 = self.square_conv(x)
        x2 = self.ver_conv(x)
        x3 = self.hor_conv(x)
        x = torch.cat([x1,x2,x3], dim=1)
        return x

class D_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(D_AC, self).__init__()
        assert isinstance(kernel_size, int) # only support square kernels
        self.square_conv = deformable_conv(in_channels, int(out_channels*0.5), kernel_size, padding, bias=False)
        self.ver_conv = nn.Conv2d(in_channels, int(out_channels*0.33), (kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)
        self.hor_conv = nn.Conv2d(in_channels, int(out_channels*0.167), (1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x1 = self.square_conv(x)
        x2 = self.ver_conv(x)
        x3 = self.hor_conv(x)
        x = torch.cat([x1,x2,x3], dim=1)
        return x

class AD_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(AD_AC, self).__init__()
        assert isinstance(kernel_size, int) # only support square kernels
        self.square_conv = Adaptive_deformable_conv(in_channels, int(out_channels*0.5), kernel_size, padding, bias=False)
        self.ver_conv = nn.Conv2d(in_channels, int(out_channels*0.33), (kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)
        self.hor_conv = nn.Conv2d(in_channels, int(out_channels*0.167), (1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x1 = self.square_conv(x)
        x2 = self.ver_conv(x)
        x3 = self.hor_conv(x)
        x = torch.cat([x1,x2,x3], dim=1)
        return x