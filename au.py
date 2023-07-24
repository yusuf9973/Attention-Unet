import torch
import torch.nn as nn
import torch.nn.functional as F

class HalfNormal(object):
    def __init__(self, scale, seed, device):
        self.scale = scale
        self.device = device
        torch.manual_seed(seed)
    def sample(self, sample_shape=torch.Size()):
        result = torch.zeros(sample_shape).to(self.device)
        result = result.normal_(mean=0, std=self.scale).abs()
        return result
class GFDConv(nn.Module):
    def __init__(self, in_features,out_features, bias=False, scale=0.1, device='cpu', seed=307, fix_w=0, fix_h=0):
        super(GFDConv, self).__init__()
        self.conv = nn.Conv2d(9 * in_features, out_features, 1, bias=bias)
        self.dis = HalfNormal(scale, seed, device)
        self.size = None
        self.device = device
        self.direction_basis = torch.tensor([[-1, 1, -1, 1, -1, 1, 0, 0],
                                             [-1, -1, 1, 1, 0, 0, -1, 1]]).float().view(-1).to(self.device)
        if fix_w != 0 and fix_h != 0:
            yy = torch.linspace(0, fix_h - 1, steps=fix_h).unsqueeze(1).repeat(1, fix_w).unsqueeze(-1).to(self.device)
            xx = torch.linspace(0, fix_w - 1, steps=fix_w).unsqueeze(0).repeat(fix_h, 1).unsqueeze(-1).to(self.device)
            self.base_coor = torch.cat([xx.repeat(1, 1, 8), yy.repeat(1, 1, 8)], dim=-1).to(self.device)
            self.size = torch.tensor([fix_w] * 8 + [fix_h] * 8).float().to(self.device)
    def forward(self, feat):
        sample_coor = self.sample_process(feat.size(2), feat.size(3))
        sample_coor_x = sample_coor[:, :, :8]
        sample_coor_y = sample_coor[:, :, 8:]
        feat = F.pad(feat, [1, 1, 1, 1]).to(sample_coor.device)
        offset_feat = feat[:, :, sample_coor_y, sample_coor_x]
        offset_feat = F.pad(offset_feat.permute(0, 4, 1, 2, 3).contiguous()
                            .view(offset_feat.size(0), -1, offset_feat.size(2), offset_feat.size(3)), [1, 1, 1, 1])
        feat = torch.cat([feat, offset_feat], dim=1)
        feat = self.conv.to(feat.device)(feat)[:, :, 1:-1, 1:-1]
        return feat
    def sample_process(self, h, w):
        if self.size is None:
            yy = torch.linspace(0, h - 1, steps=h).unsqueeze(1).repeat(1, w).unsqueeze(-1).to(self.device)
            xx = torch.linspace(0, w - 1, steps=w).unsqueeze(0).repeat(h, 1).unsqueeze(-1).to(self.device)
            base_coor = torch.cat([xx.repeat(1, 1, 8), yy.repeat(1, 1, 8)], dim=-1).to(self.device)
            size = torch.tensor([w] * 8 + [h] * 8).float().to(self.device)
        else:
            size = self.size
            base_coor = self.base_coor
        sample_ = self.dis.sample(torch.Size([h, w, 16]))
        offset = sample_ * self.direction_basis * size
        sample_coor = base_coor + offset
        sample_coor[:, :, :8] = torch.clamp(sample_coor[:, :, :8], min=0, max=w - 1)
        sample_coor[:, :, 8:] = torch.clamp(sample_coor[:, :, 8:], min=0, max=h - 1)
        return (sample_coor + 1).long()
def gaussian_dynamic_conv(in_channels, out_channels, kernel_size, padding, bias=False):
    fix_w = 0
    fix_h = 0
    seed = 307
    scale = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return GFDConv(in_channels, out_channels, bias, scale, device, seed, fix_w, fix_h)



class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,index=0):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            gaussian_dynamic_conv(in_ch, out_ch, kernel_size=3,  padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            gaussian_dynamic_conv(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
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
    def __init__(self, n_channels, n_classes, bilinear=False,index = 0):
        super(UNet, self).__init__()
        img_ch=n_channels 
        output_ch=n_classes 
        self.index=index
        n1=64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, filters[0],index)
        self.Conv2 = conv_block(filters[0], filters[1],index)
        self.Conv3 = conv_block(filters[1], filters[2],index)
        self.Conv4 = conv_block(filters[2], filters[3],index)
        self.Conv5 = conv_block(filters[3], filters[4],index)
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3],index)
        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2],index)
        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1],index)
        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0],index)
        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
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
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)
        return out
device = 'cuda'
model = UNet(n_channels=3,n_classes=6,bilinear=False,index=0)
input_tensor = torch.randn(1, 3, 512, 512)
model.to(device)
input_tensor.to(device)
model.eval()
output_tensor = model(input_tensor)
print("Output shape:", output_tensor.shape)
