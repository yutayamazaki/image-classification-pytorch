import unittest

import torch.nn as nn

import utils


class CountParametersTest(unittest.TestCase):

    def test_linear(self):
        net: nn.Module = nn.Linear(2, 1)
        num_params: int = utils.count_parameters(net)

        self.assertEqual(num_params, 3)  # {w: 2, b: 1}
