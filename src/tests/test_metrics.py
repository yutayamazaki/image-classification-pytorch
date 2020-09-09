import unittest

import torch

import metrics


class AccuracyTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_accuracy_1d(self):
        y_true = torch.Tensor([0, 1, 0, 2])
        y_pred = torch.Tensor([0, 0, 0, 2])
        calculated_acc = metrics.accuracy_score(y_true, y_pred)
        collect_acc = 0.75
        self.assertAlmostEqual(calculated_acc, collect_acc)

    def test_accuracy_2d(self):
        y_true = torch.Tensor([0, 1, 0, 2])
        y_pred = torch.Tensor([
            [0.7, 0.2, 0.1],
            [0.8, 0.2, 0.0],
            [0.8, 0.1, 0.1],
            [0.3, 0.2, 0.8],
        ])
        calculated_acc = metrics.accuracy_score(y_true, y_pred)
        collect_acc = 0.75
        self.assertAlmostEqual(calculated_acc, collect_acc)

    def test_value_error_3d(self):
        y_true = torch.Tensor([0, 1, 0, 2])
        y_pred = torch.zeros((4, 2, 1))

        with self.assertRaises(ValueError):
            metrics.accuracy_score(y_true, y_pred)

    def test_value_error_inconsistent_length(self):
        y_true = torch.Tensor([0, 1])
        y_pred = torch.Tensor([0])

        with self.assertRaises(ValueError):
            metrics.accuracy_score(y_true, y_pred)
