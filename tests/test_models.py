import unittest

import torch

from src import models


class UnetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.in_channels = 3
        self.num_classes = 10
        self.h = 64
        self.w = 64
        self.b = 4
        self.unet = models.UNet(self.in_channels, self.num_classes)

    def test_output_shape(self):
        x = torch.zeros((self.b, self.in_channels, self.h, self.w))
        outputs = self.unet(x)
        collect_size = torch.Size((self.b, self.num_classes, self.h, self.w))
        self.assertEqual(outputs.size(), collect_size)
