import torch
import torch.nn as nn


class Swish(nn.Module):
    """ Siwsh activation: https://arxiv.org/abs/1710.05941

    Parameters
    ----------
    beta: float
        beta: x * sigmoid(βx)

    inplace: bool
        inplace or not.
    """
    def __init__(self, beta=1.0, inplace=True):
        super(Swish, self).__init__()
        self.beta = beta
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(self.beta * torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(self.beta * x)
