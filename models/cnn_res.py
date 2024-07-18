import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_sphere import *
from model.conv import MBConv
from models.wideresnet import WideResNet

debug = False

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class GCA(nn.Module):
    def __init__(self, in_planes, feature_size,groups=32):
        super(GBAM, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.ch_AvgPool = nn.AvgPool3d(feature_size, feature_size)
        self.ch_MaxPool = nn.MaxPool3d(feature_size, feature_size)
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Softmax = nn.Softmax(1)

        self.id_conv1 = make_conv3d(in_planes, feature_size, kernel_size=3, stride=1, padding=1)
        self.pool_h = nn.AdaptiveAvgPool3d((None, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_t = nn.AdaptiveAvgPool3d((None, 1, 1))
        mip = max(8, in_planes // groups)
        self.conv1 = nn.Conv3d(in_planes, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.conv2 = nn.Conv3d(mip, in_planes, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv3d(mip, in_planes, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.relu1=nn.ReLU(inplace=True)
        self.relu2=nn.Sigmoid()
    def forward(self, x):
        # print("-------------------input-----------------", x.size())
        x_permute = x.permute(0, 1, 3, 4, 2)
        x_ch_avg_pool = self.ch_AvgPool(x_permute).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x_permute).view(x.size(0), -1)
        x_ch_avg_linear = self.ch_Linear2(self.relu1(self.ch_Linear1(x_ch_avg_pool)))
        x_ch_max_linear = self.ch_Linear2(self.relu1(self.ch_Linear1(x_ch_max_pool)))
        # ch_out = self.ch_Softmax(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)
        ch_out = (x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)
        ch_out = ch_out.permute(0, 1, 3, 4, 2)
        ch_out = ch_out * x
        # print("-------------------output-----------------", ch_out.size())
        n, c, t, h, w, = ch_out.size()
        x_t = self.pool_t(ch_out).sigmoid()
        x_h = self.pool_h(ch_out)
        x_w = self.pool_w(ch_out).permute(0, 1, 2, 4, 3)  # 0132
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 2, 4, 3)  # 0132
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, -1, h, w)
        # print("-------------------output-----------------", x_h.size())
        x_w = x_w.expand(-1, -1, -1, h, w)
        # print("-------------------output-----------------", x_w.size())
        y = self.relu1(x_t * x_w * x_h)
        out = y * x
        # print("-------------------output-----------------", out.size())
        return out


def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: my conv3d module
    """
    module = nn.Sequential(

        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=groups,
                  bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module


def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, groups=1,
                     bias=True):
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


def conv3d_pooling(in_channels, kernel_size, stride=1,
                   dilation=1, groups=1,
                   bias=False):
    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)



class ConvRes(nn.Module):
    def __init__(self, config):
        super(ConvRes, self).__init__()
        self.conv1 = WideResNet(in_channels=1, out_channels=4)
        self.conv2 = WideResNet(in_channels=4, out_channels=4)
        self.config = config
        self.last_channel = 4
        self.first_gca = GCA(4, 32)
        layers = []
        i = 0
        for stage in config:
            i = i+1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(WideResNet(self.last_channel, channel))
                self.last_channel = channel
            layers.append(GBAM(self.last_channel, 32//(2**i)))

        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=4, stride=4)
        self.fc = AngleLinear(in_features=self.last_channel, out_features=2)

    def forward(self, inputs):
        if debug:
            print(inputs.size())
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        out = self.first_gca(out)
        out = self.layers(out)
        if debug:
            print(out.size())
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        if debug:
            print(out.size())
        out = self.fc(out)
        return out

def test():
    global debug
    debug = True
    net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    inputs = torch.randn((1, 1, 32, 32, 32))
    output = net(inputs)
    print(net.config)
    print(output)
