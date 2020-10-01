# Based on this repo
# https://github.com/ronghuaiyang/arcface-pytorch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance
        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            s (float): norm of input feature
            m (float): margin
            easy_margin (bool):
            cos(theta + m)
        """
    def __init__(
        self, in_features: int, out_features: int, s: float = 30.0,
        m: float = 0.50, easy_margin=False
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        """
        Args:
            inputs (torch.Tensor): (B, in_features).
            labels (torch.Tensor): (B, ).
        Returns:
            torch.Tensor: (B, out_features)
        """
        # --------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------- convert label to one-hot ---------------------
        device: torch.device = inputs.device
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------torch.where(out_i = {x_i if condition_i else y_i) -------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        s (float): norm of input feature
        m (float): margin
        cos(theta) - m
    """

    def __init__(
        self, in_features: int, out_features: int, s: float = 30.0,
        m: float = 0.40
    ):
        super(AddMarginProduct, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.s: float = s
        self.m: float = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels):
        """
        Args:
            inputs (torch.Tensor): (B, in_features).
            labels (torch.Tensor): (B, ).
        Returns:
            torch.Tensor: (B, out_features)
        """
        # --------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------- convert label to one-hot ---------------------
        device: torch.device = inputs.device
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------torch.where(out_i = {x_i if condition_i else y_i) -------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        m (int): margin
        cos(m*theta)
    """
    def __init__(self, in_features: int, out_features: int, m: int = 4):
        super(SphereProduct, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.m: int = m
        self.base: float = 1000.0
        self.gamma: float = 0.12
        self.power: int = 1
        self.LambdaMin: float = 5.0
        self.iter: int = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, inputs, labels):
        self.iter += 1
        self.lamb = max(
            self.LambdaMin, self.base * (
                1 + self.gamma * self.iter
            ) ** (-1 * self.power)
        )

        # --------------------- cos(theta) & phi(theta) ---------------------
        cos_theta = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(inputs, 2, 1)

        # --------------------- convert label to one-hot ---------------------
        device: torch.device = inputs.device
        one_hot = torch.zeros(cos_theta.size(), device=device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # --------------------- Calculate output ---------------------
        output = (
            one_hot * (phi_theta - cos_theta) / (1 + self.lamb)
        ) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
