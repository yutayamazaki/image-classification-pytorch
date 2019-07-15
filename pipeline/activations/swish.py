import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, sigma=1.0, inplace=True):
        super(Swish, self).__init__()
        self.sigma = sigma
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(self.sigma * torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(self.sigma * x)