import unittest

import torch

import metrics


class AccuracyScoreTest(unittest.TestCase):

    def test_accuracy_1d(self):
        outputs: torch.Tensor = torch.Tensor([[0.9, 0.1], [0.9, 0.1]])
        targets: torch.Tensor = torch.Tensor((0, 1))
        acc: float = metrics.accuracy_score(outputs, targets)
        self.assertAlmostEqual(acc, 0.5)
