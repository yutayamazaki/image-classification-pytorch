import unittest

import torch
import torch.nn as nn
import torchvision

import layers


class ArcMargiProductTests(unittest.TestCase):

    def test_forward(self):
        net: nn.Module = torchvision.models.alexnet(pretrained=False)
        metric_fc: nn.Module = layers.ArcMarginProduct(1000, 10)
        x: torch.Tensor = torch.randn((2, 3, 128, 128))
        labels: torch.Tensor = torch.Tensor([0, 1])

        out = net(x)
        self.assertEqual(out.size(), torch.Size((2, 1000)))
        out = metric_fc(out, labels)
        self.assertEqual(out.size(), torch.Size((2, 10)))


class AddMargiProductTests(unittest.TestCase):

    def test_forward(self):
        net: nn.Module = torchvision.models.alexnet(pretrained=False)
        metric_fc: nn.Module = layers.AddMarginProduct(1000, 10)
        x: torch.Tensor = torch.randn((2, 3, 128, 128))
        labels: torch.Tensor = torch.Tensor([0, 1])

        out = net(x)
        self.assertEqual(out.size(), torch.Size((2, 1000)))
        out = metric_fc(out, labels)
        self.assertEqual(out.size(), torch.Size((2, 10)))


class SphereProductTests(unittest.TestCase):

    def test_forward(self):
        net: nn.Module = torchvision.models.alexnet(pretrained=False)
        metric_fc: nn.Module = layers.SphereProduct(1000, 10)
        x: torch.Tensor = torch.randn((2, 3, 128, 128))
        labels: torch.Tensor = torch.Tensor([0, 1])

        out = net(x)
        self.assertEqual(out.size(), torch.Size((2, 1000)))
        out = metric_fc(out, labels)
        self.assertEqual(out.size(), torch.Size((2, 10)))
