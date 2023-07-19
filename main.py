#torch implementation
import torch.nn as nn
import torch
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

class conv_block_nested(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,index=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(

            gaussian_dynamic_conv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            gaussian_dynamic_conv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )        
    def forward(self, inputs):
        return self.conv(inputs)

class UNet(nn.Module):
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

inp = torch.randn((1, 3, 256, 256))
model = UNet(n_channels=3, n_classes=5, bilinear=False,index = 0)
out = model(inp)
print(out.shape)
