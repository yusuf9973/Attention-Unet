import torch.nn as nn

class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ACBlock, self).__init__()
        assert isinstance(kernel_size, int) # only support square kernels
        self.square_conv = nn.Conv2d(in_channels, int(out_channels*0.5), (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.ver_conv = nn.Conv2d(in_channels, int(out_channels*0.33), (kernel_size, 1), stride=stride, padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)
        self.hor_conv = nn.Conv2d(in_channels, int(out_channels*0.167), (1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x1 = self.square_conv(x)
        x2 = self.ver_conv(x)
        x3 = self.hor_conv(x)
        x = x1 + x2 + x3
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F



class conv_block_nested(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,index=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(

            ACBlock(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ACBlock(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )        
    def forward(self, inputs):
        return self.conv(inputs)

class UNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, n_channels, n_classes, bilinear=False,index = 0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.index = index

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(n_channels, filters[0], filters[0],index)
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1],index)
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2],index)
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3],index)
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4],index)

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0],index)
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1],index)
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2],index)
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3],index)

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0],index)
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1],index)
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2],index)

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0],index)
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1],index)

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0],index)

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


    def use_checkpointing(self):
        self.conv0_0 = torch.utils.checkpoint(self.conv0_0)
        self.conv1_0 = torch.utils.checkpoint(self.conv1_0)
        self.conv0_1 = torch.utils.checkpoint(self.conv0_1)

        self.conv2_0 = torch.utils.checkpoint(self.conv2_0)
        self.conv1_1 = torch.utils.checkpoint(self.conv1_1)
        self.conv0_2 = torch.utils.checkpoint(self.conv0_2)

        self.conv3_0 = torch.utils.checkpoint(self.conv3_0)
        self.conv2_1 = torch.utils.checkpoint(self.conv2_1)
        self.conv1_2 = torch.utils.checkpoint(self.conv1_2)
        self.conv0_3 = torch.utils.checkpoint(self.conv0_3)

        self.conv4_0 = torch.utils.checkpoint(self.conv4_0)
        self.conv3_1 = torch.utils.checkpoint(self.conv3_1)
        self.conv2_2 = torch.utils.checkpoint(self.conv2_2)
        self.conv1_3 = torch.utils.checkpoint(self.conv1_3)
        self.conv0_4 = torch.utils.checkpoint(self.conv0_4)

        self.final = torch.utils.checkpoint(self.final)

input = torch.randn(1,3,256,256)
model = UNet(3,5,False,0)
output = UNet(input)
print(output.shape)
