import torch
import torch.nn as nn
def conv3x3(x, out_filters, strides=(1, 1)):
    x = torch.nn.Conv2d(x.shape[1], out_filters, 3, padding='same', stride=strides, bias=False)(x)
    return x
def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)
    x = torch.nn.ReLU()(x)

    x = conv3x3(x, out_filters)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)

    if with_conv_shortcut:
        residual = torch.nn.Conv2d(input.shape[1], out_filters, 1, stride=strides, bias=False)(input)
        residual = torch.nn.BatchNorm2d(residual.shape[1])(residual)
        x = torch.add(x, residual)
    else:
        x = torch.add(x, input)

    x = torch.nn.ReLU()(x)
    return x
def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = torch.nn.Conv2d(input.shape[1], de_filters, 1, bias=False)(input)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)
    x = torch.nn.ReLU()(x)

    x = torch.nn.Conv2d(x.shape[1], de_filters, 3, padding='same', stride=strides, bias=False)(x)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)
    x = torch.nn.ReLU()(x)

    x = torch.nn.Conv2d(x.shape[1], out_filters, 1, bias=False)(x)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)

    if with_conv_shortcut:
        residual = torch.nn.Conv2d(input.shape[1], out_filters, 1, stride=strides, bias=False)(input)
        residual = torch.nn.BatchNorm2d(residual.shape[1])(residual)
        x = torch.add(x, residual)
    else:
        x = torch.add(x, input)

    x = torch.nn.ReLU()(x)
    return x
def stem_net(input):
    x = torch.nn.Conv2d(input.shape[1], 64, 3, padding='same', stride=(2, 2), bias=False)(input)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)
    x = torch.nn.ReLU()(x)

    x = torch.nn.Conv2d(x.shape[1], 64, 3, padding='same', stride=(2, 2), bias=False)(x)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)
    x = torch.nn.ReLU()(x)

    x = bottleneck_Block(x, 256, with_conv_shortcut=True)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)

    return x
def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = torch.nn.Conv2d(x.shape[1], out_filters_list[0], 3, padding='same', bias=False)(x)
    x0 = torch.nn.BatchNorm2d(x0.shape[1])(x0)
    x0 = torch.nn.ReLU()(x0)

    x1 = torch.nn.Conv2d(x.shape[1], out_filters_list[1], 3, padding='same', stride=(2, 2), bias=False)(x)
    x1 = torch.nn.BatchNorm2d(x1.shape[1])(x1)
    x1 = torch.nn.ReLU()(x1)

    return [x0, x1]
def make_branch1_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def make_branch1_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = torch.nn.Conv2d(x[1].shape[1], 32, 1, bias=False)(x[1])
    x0_1 = torch.nn.BatchNorm2d(x0_1.shape[1])(x0_1)
    x0_1 = torch.nn.Upsample(scale_factor=(2, 2))(x0_1)
    x0 = torch.add(x0_0, x0_1)

    x1_0 = torch.nn.Conv2d(x[0].shape[1], 64, 3, padding='same', stride=(2, 2), bias=False)(x[0])
    x1_0 = torch.nn.BatchNorm2d(x1_0.shape[1])(x1_0)
    x1_1 = x[1]
    x1 = torch.add(x1_0, x1_1)
    return [x0, x1]
def transition_layer2(x, out_filters_list=[32, 64, 128]):
    x0 = torch.nn.Conv2d(x[0].shape[1], out_filters_list[0], 3, padding='same', bias=False)(x[0])
    x0 = torch.nn.BatchNorm2d(x0.shape[1])(x0)
    x0 = torch.nn.ReLU()(x0)

    x1 = torch.nn.Conv2d(x[1].shape[1], out_filters_list[1], 3, padding='same', bias=False)(x[1])
    x1 = torch.nn.BatchNorm2d(x1.shape[1])(x1)
    x1 = torch.nn.ReLU()(x1)

    x2 = torch.nn.Conv2d(x[1].shape[1], out_filters_list[2], 3, padding='same', stride=(2, 2), bias=False)(x[1])
    x2 = torch.nn.BatchNorm2d(x2.shape[1])(x2)
    x2 = torch.nn.ReLU()(x2)

    return [x0, x1, x2]
def make_branch2_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def make_branch2_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def make_branch2_2(x, out_filters=128):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = torch.nn.Conv2d(x[1].shape[1], 32, 1, bias=False)(x[1])
    x0_1 = torch.nn.BatchNorm2d(x0_1.shape[1])(x0_1)
    x0_1 = torch.nn.Upsample(scale_factor=(2, 2))(x0_1)
    x0_2 = torch.nn.Conv2d(x[2].shape[1], 32, 1, bias=False)(x[2])
    x0_2 = torch.nn.BatchNorm2d(x0_2.shape[1])(x0_2)
    x0_2 = torch.nn.Upsample(scale_factor=(4, 4))(x0_2)
    x0 = torch.add(torch.add(x0_0, x0_1), x0_2)

    x1_0 = torch.nn.Conv2d(x[0].shape[1], 64, 3, padding='same', stride=(2, 2), bias=False)(x[0])
    x1_0 = torch.nn.BatchNorm2d(x1_0.shape[1])(x1_0)
    x1_1 = x[1]
    x1_2 = torch.nn.Conv2d(x[2].shape[1], 64, 1, bias=False)(x[2])
    x1_2 = torch.nn.BatchNorm2d(x1_2.shape[1])(x1_2)
    x1_2 = torch.nn.Upsample(scale_factor=(2, 2))(x1_2)
    x1 = torch.add(torch.add(x1_0, x1_1), x1_2)

    x2_0 = torch.nn.Conv2d(x[0].shape[1], 32, 3, padding='same', stride=(2, 2), bias=False)(x[0])
    x2_0 = torch.nn.BatchNorm2d(x2_0.shape[1])(x2_0)
    x2_0 = torch.nn.ReLU()(x2_0)
    x2_0 = torch.nn.Conv2d(x2_0.shape[1], 128, 3, padding='same', stride=(2, 2), bias=False)(x2_0)
    x2_0 = torch.nn.BatchNorm2d(x2_0.shape[1])(x2_0)
    x2_1 = torch.nn.Conv2d(x[1].shape[1], 128, 3, padding='same', stride=(2, 2), bias=False)(x[1])
    x2_1 = torch.nn.BatchNorm2d(x2_1.shape[1])(x2_1)
    x2_2 = x[2]
    x2 = torch.add(torch.add(x2_0, x2_1), x2_2)

    return [x0, x1, x2]
def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x0 = torch.nn.Conv2d(x[0].shape[1], out_filters_list[0], 3, padding='same', bias=False)(x[0])
    x0 = torch.nn.BatchNorm2d(x0.shape[1])(x0)
    x0 = torch.nn.ReLU()(x0)

    x1 = torch.nn.Conv2d(x[1].shape[1], out_filters_list[1], 3, padding='same', bias=False)(x[1])
    x1 = torch.nn.BatchNorm2d(x1.shape[1])(x1)
    x1 = torch.nn.ReLU()(x1)

    x2 = torch.nn.Conv2d(x[2].shape[1], out_filters_list[2], 3, padding='same', bias=False)(x[2])
    x2 = torch.nn.BatchNorm2d(x2.shape[1])(x2)
    x2 = torch.nn.ReLU()(x2)

    x3 = torch.nn.Conv2d(x[2].shape[1], out_filters_list[3], 3, padding='same', stride=(2, 2), bias=False)(x[2])
    x3 = torch.nn.BatchNorm2d(x3.shape[1])(x3)
    x3 = torch.nn.ReLU()(x3)

    return [x0, x1, x2, x3]
def make_branch3_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def make_branch3_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def make_branch3_2(x, out_filters=128):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x
def make_branch3_3(x, out_filters=256):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x

def fuse_layer3(x):
    x0_0 = x[0]
    x0_1 = nn.Conv2d(32, 1, bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = nn.BatchNorm2d(32)(x0_1)
    x0_1 = nn.Upsample(scale_factor=2)(x0_1)
    x0_2 = nn.Conv2d(32, 1, bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = nn.BatchNorm2d(32)(x0_2)
    x0_2 = nn.Upsample(scale_factor=4)(x0_2)
    x0_3 = nn.Conv2d(32, 1, bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = nn.BatchNorm2d(32)(x0_3)
    x0_3 = nn.Upsample(scale_factor=8)(x0_3)
    x0 = torch.cat([x0_0, x0_1, x0_2, x0_3], dim=-1)
    return x0
def final_layer(x, classes=1):
    x = torch.nn.Upsample(scale_factor=(32, 32))(x)
    x = torch.nn.Conv2d(x.shape[1], classes, 1, bias=False)(x)
    x = torch.nn.BatchNorm2d(x.shape[1])(x)
    x = torch.nn.Sigmoid()(x)
    x = torch.nn.functional.interpolate(x, size=(256, 256))
    return x

def seg_hrnet(inputs, classes):
    x = stem_net(inputs)
    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])
    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])
    x = transition_layer3(x)
    x0 = make_branch3_0(x[0])
    x1 = make_branch3_1(x[1])
    x2 = make_branch3_2(x[2])
    x3 = make_branch3_3(x[3])
    x = fuse_layer3([x0, x1, x2, x3])

    out = final_layer(x, classes=classes)

    model = nn.Module(inputs=inputs, outputs=out)

    return model


input_tensor = torch.randn(1, 3, 256, 256)
output_tensor = seg_hrnet(input_tensor.shape,5)
print(output_tensor.shape)
