import os
import unittest

import numpy as np
import torch
import torchvision

import visualize as vis


class PlotConfusionMatrixTests(unittest.TestCase):

    def setUp(self):
        self.save_path = 'fig.png'

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_plot(self):
        conf_mat: np.ndarray = np.array([[1, 0], [1, 2]])
        vis.plot_confusion_matrix(conf_mat, self.save_path)

        self.assertTrue(os.path.exists(self.save_path))


class GradCAMTests(unittest.TestCase):

    def test_call(self):
        gradcam = vis.GradCAM(
            torchvision.models.resnet18(False),
            ['layer4']
        )
        inputs = torch.randn((1, 3, 224, 224))
        out = gradcam(inputs)

        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (224, 224))


class ApplyGradCAMOnImageTests(unittest.TestCase):

    def test_return_shape(self):
        img: np.ndarray = np.zeros((10, 10, 3))
        mask: np.ndarray = np.zeros((10, 10))

        out = vis.apply_gradcam_on_image(img, mask)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (10, 10, 3))
