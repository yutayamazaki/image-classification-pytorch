import unittest

import numpy as np
import torch
import torchvision

from pipeline import samplers


class ImbalancedSamplerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.size = 100
        self.x_dim = 10
        X = torch.zeros((self.size, self.x_dim))
        y = torch.zeros(self.size)

        dataset = torch.utils.data.TensorDataset(X, y)
        sampler = samplers.ImbalancedSampler(dataset)
        self.batch_size = 16
        self.loader = torch.utils.data.DataLoader(
                      dataset,
                      sampler=sampler,
                      batch_size=self.batch_size
        )

    def test_shape(self):
        X, y = iter(self.loader).next()
        self.assertEqual(list(X.size()), [self.batch_size, self.x_dim])
        self.assertEqual(list(y.size()), [self.batch_size])