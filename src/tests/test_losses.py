import unittest

import torch

import losses


class FocalLossTest(unittest.TestCase):

    def setUp(self) -> None:
        self.criterion = losses.FocalLoss()

    def test_loss(self):
        targets: torch.Tensor = torch.Tensor([0, 0, 1]).long()
        outputs: torch.Tensor = torch.Tensor([
            [1.0, 0.0], [1.0, 0.0], [0.5, 0.5]
        ])

        loss: torch.Tensor = self.criterion(outputs, targets)
        self.assertEqual(loss.size(), torch.Size(()))
