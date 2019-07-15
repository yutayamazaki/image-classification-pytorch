import torch
import torch.nn as nn


class cSE(nn.Module):
    def __init__(self, n_channels):
        super(cSE, self).__init__()
        self.squeeze = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.squeeze(x)
        y = self.sigmoid(x)
        return x * y


class sSE(nn.Module):
    def __init__(self, n_channels, reduction=4):
        super(sSE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(n_channels, n_channels//reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(n_channels//reduction, n_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmodi = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = self.relu(self.conv1(y))
        y = self.sigmoid(self.conv2(y))
        return x * y


class scSE(nn.Module):
    def __init__(self, n_channels, reduction=4):
        super(scSE, self).__init__()
        self.sse = sSE(n_channels, reduction=reduction)
        self.cse = cSE(n_channels)

    def foward(self, x):
        out = self.sse(x) + self.cse(x)
        return out