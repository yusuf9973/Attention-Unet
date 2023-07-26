from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.conv_weight_m = torch.ones([outc,inc,kernel_size,kernel_size], requires_grad=False).cuda()
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0.5)
            self.m_conv.register_backward_hook(self._set_lr)
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset

def Adaptive_deformable_conv(in_channels, out_channels, kernel_size, padding, bias=False):
	stride=1
	modulation=False
	bias = None
	return DeformConv2d(in_channels,out_channels,kernel_size,padding,stride,bias,modulation)

import torch
import torch.nn as nn
import torch.nn.functional as F
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,index=0):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            Adaptive_deformable_conv(in_ch, out_ch, kernel_size=3,  padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Adaptive_deformable_conv(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
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
class AttenUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,index = 0):
        super(AttenUNet, self).__init__()
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


#train code:


import matplotlib.pyplot as plt
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_optimizer as optim
from torch.optim import Adam
from codecarbon.emissions_tracker import EmissionsTracker
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

time_list = []
torch.manual_seed(42)
def train(x_train,y_train,n_classes,names,d,batch_size,num_epochs,n_channels):
    device = 'cuda'
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    train_dataset = CustomDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    class_counts = torch.zeros(n_classes)
    for i in range(n_classes):
        class_counts[i] = (y_train == i).sum()
    total_count = class_counts.sum()
    class_weights = total_count / class_counts
    class_weights = class_weights / class_weights.sum()
    for q in range(len(names)):
          class_weights = class_weights.to(device)
          tracker = EmissionsTracker(save_to_file=True, output_file='data/'+names[q]+d+'_my_emissions.csv', log_level="ERROR")
          tracker.start()
          t1 = time.time()
          model = AttenUNet(n_channels, n_classes, bilinear=False,index=q)
          model.to(device)
          print(names[q])
          train_losses = []
          train_accs = []
          val_losses = []
          val_accs = []
          learning_rate = 0.0001
          criterion = nn.CrossEntropyLoss(weight=class_weights)
          l_t = len(train_dataloader)
          l_v = len(val_dataloader)
          best_acc = 0.0
          best_epoch = 0
          early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.01, path='model/es/' + names[q] + d+'.pth')
          optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
          for epoch in range(num_epochs):
              running_loss = 0.0
              running_acc = 0.0
              t = 0
              v = 0
              model.train()
              for i, (inputs, labels) in enumerate(train_dataloader):
                  inputs = inputs.permute(0, 3, 1, 2)
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  optimizer.zero_grad()
                  outputs = model(inputs)
                  outputs = torch.exp(outputs)
                  loss = criterion(outputs, labels)
                  acc = (outputs.argmax(dim=1) == labels).float().mean()
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  running_acc += acc.item()
                  t+=1
                  print(f'\rEpoch {epoch+1}: Train Progress: {int((t/l_t)*100)}%',end="")
              train_losses.append(running_loss / len(train_dataloader))
              train_accs.append(running_acc / len(train_dataloader))
              val_loss = 0.0
              val_acc = 0.0
              model.eval()
              with torch.no_grad():
                  for i ,(inputs, labels) in enumerate(val_dataloader):
                      inputs = inputs.permute(0, 3, 1, 2)
                      inputs = inputs.to(device)
                      labels = labels.to(device)
                      outputs = model(inputs)
                      outputs = torch.exp(outputs)
                      loss = criterion(outputs, labels)
                      acc = (outputs.argmax(dim=1) == labels).float().mean()
                      val_loss += loss.item()
                      val_acc += acc.item()
                      v+=1
                      print(f'\rEpoch {epoch+1}: Train Progress: {int((t/l_t)*100)}% Validation Progress: {int((v/l_v)*100)}%  ',end="")
              val_losses.append(val_loss / len(val_dataloader))
              val_accs.append(val_acc / len(val_dataloader))
              scheduler.step(val_loss)
              print(f'\rEpoch {epoch+1}: Training Loss: {running_loss / len(train_dataloader):.4f}, Training Accuracy: {running_acc / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {val_acc / len(val_dataloader):.4f}')
              if val_acc > best_acc:
                  best_acc = val_acc
                  torch.save(model.state_dict(),'model/'+names[q]+d+'.pth')
              early_stopping(val_loss, model)
              if early_stopping.early_stop:
                  print(f'Early stopping at epoch {epoch+1}')
                  break
names = ['AttenUnet_ADC']
d = '1'
y_train = np.load('dataset'+d+'/y_train.npz')['arr_0'][:4]
x_train = np.load('dataset'+d+'/x_train.npz')['arr_0'][:4]
x_test = np.load('dataset'+d+'/x_test.npz')['arr_0'][:2]
y_test = np.load('dataset'+d+'/y_test.npz')['arr_0'][:2]
print("Training...")
train(x_train,y_train,5,names,d,1,1,3)
