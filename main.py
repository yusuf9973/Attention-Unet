import torch
import torch.nn as nn
import torch.nn.functional as F
conv_function = [conv2d, SpatialSeparableConv2d, ACBlock, gaussian_dynamic_conv, deformable_conv,Adaptive_deformable_conv,GD_AC,D_AC,AD_AC]


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,index=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(

            conv_function[index](in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_function[index](mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True,index=0):
        super().__init__()
        if bilinear:
            self.index=index
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_block(in_channels, out_channels, in_channels // 2,index)
        else:
            self.index=index
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_block(in_channels, out_channels,None,index)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self,n_channels=3, n_classes=5, bilinear=False,index=0):
        super(UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.index = index



        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0],None,index)
        self.Conv2 = conv_block(filters[0], filters[1],None,index)
        self.Conv3 = conv_block(filters[1], filters[2],None,index)
        self.Conv4 = conv_block(filters[2], filters[3],None,index)
        self.Conv5 = conv_block(filters[3], filters[4],None,index)

        self.Up5 = up_conv(filters[4], filters[3],True,index)
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3],None,index)

        self.Up4 = up_conv(filters[3], filters[2],True,index)
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2],None,index)

        self.Up3 = up_conv(filters[2], filters[1],True,index)
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1],None,index)

        self.Up2 = up_conv(filters[1], filters[0],True,index)
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0],None,index)

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5,e4)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,e3)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,e2)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,e1)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out


    def use_checkpointing(self):
        self.Conv1 = torch.utils.checkpoint(self.Conv1)
        self.Maxpool1 = torch.utils.checkpoint(self.Maxpool1)
        self.Conv2 = torch.utils.checkpoint(self.Conv2)
        self.Maxpool2 = torch.utils.checkpoint(self.Maxpool2)
        self.Conv3 = torch.utils.checkpoint(self.Conv3)
        self.Maxpool3 = torch.utils.checkpoint(self.Maxpool3)
        self.Conv4 = torch.utils.checkpoint(self.Conv4)
        self.Maxpool4 = torch.utils.checkpoint(self.Maxpool4)
        self.Conv5 = torch.utils.checkpoint(self.Conv5)	
        self.Up5 = torch.utils.checkpoint(self.Up5)
        self.Att5 = torch.utils.checkpoint(self.Att5)
        self.Up_conv5 = torch.utils.checkpoint(self.Up_conv5)
        self.Up4 = torch.utils.checkpoint(self.Up4)
        self.Att4 = torch.utils.checkpoint(self.Att4)
        self.Up_conv4 = torch.utils.checkpoint(self.Up_conv4)
        self.Up3 = torch.utils.checkpoint(self.Up3)
        self.Att3 = torch.utils.checkpoint(self.Att3)
        self.Up_conv3 = torch.utils.checkpoint(self.Up_conv3)
        self.Up2 = torch.utils.checkpoint(self.Up2)
        self.Att2 = torch.utils.checkpoint(self.Att2)
        self.Up_conv2 = torch.utils.checkpoint(self.Up_conv2)
        self.Conv = torch.utils.checkpoint(self.Conv)

model = UNet(n_channels=3, n_classes=5, bilinear=False,index=0)
input = torch.randn(1,3,256,256)
output = model(input)
print(output.shape) 
