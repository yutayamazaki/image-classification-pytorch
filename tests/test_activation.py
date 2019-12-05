import unittest

import numpy as np
import torch

from src import activations


class SwishTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_func(self):
        beta = 2.0
        inplace = False
        swish = activations.Swish(beta=beta, inplace=inplace)

        x = torch.Tensor([-1, 0, 1])
        y = swish(x)
        y_collect = x * torch.sigmoid(beta * x)

        y = y.numpy()
        y_collect = y_collect.numpy()
        np.testing.assert_almost_equal(y, y_collect)

    def test_inplace(self):
        swish = activations.Swish(inplace=True)

        x = torch.Tensor([-1, 0, 1])
        y = swish(x)

        x = x.numpy()
        y = y.numpy()

        np.testing.assert_almost_equal(x, y)
