import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0 ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        hidden_dim = int(in_planes * 4)
        self.conv = nn.Sequential(
            # pw
            # down-sample in the first conv
            nn.Conv3d(in_planes, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            # dw
            nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),

            nn.Conv3d(hidden_dim, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_planes),
        )


    def forward(self, x):

        out = self.conv(x if self.equalInOut else x)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, in_channels, out_channels, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [out_channels, out_channels*widen_factor, out_channels*widen_factor, out_channels*widen_factor]

        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv3d(in_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(1, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(1, nChannels[1], nChannels[2], block, 1, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(1, nChannels[2], nChannels[3], block, 1, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = out_channels

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # print("input========================", x.size())
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out
