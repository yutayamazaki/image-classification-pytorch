import unittest

import torch

from pipeline import losses


class FocalLossTest(unittest.TestCase):

    def setUp(self) -> None:
        self.criterion = losses.FocalLoss()

    def tearDown(self) -> None:
        pass

    def test_loss(self):
        y_true = torch.Tensor([0, 0, 1])
        y_pred = torch.Tensor([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])

        loss = self.criterion(y_true, y_pred)
        self.assertAlmostEqual(loss.item(), 0.299271047)

    def test_value_error_y_true_dim(self):
        y_true = torch.Tensor([[0.9, 0.1], [0.0, 1.0]])

        with self.assertRaises(ValueError):
            self.criterion(y_true, y_true)

    def test_value_error_y_pred_dim(self):
        y_pred = torch.Tensor([0, 0, 10])

        with self.assertRaises(ValueError):
            self.criterion(y_pred, y_pred)