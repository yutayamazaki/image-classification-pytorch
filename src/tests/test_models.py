import unittest
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torchvision

import models


class LoadResNetTests(unittest.TestCase):

    def test_load_resnet_models(self):
        attributes: Tuple[str, ...] = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        )
        for name in attributes:
            net = models._load_resnet(name)
            self.assertIsInstance(net, torchvision.models.ResNet)

    def test_raise(self):
        name: str = 'invalid-resnet'
        with self.assertRaises(ValueError):
            models._load_resnet(name)


class CosineSimilarityTests(unittest.TestCase):

    def test_return(self):
        feats = torch.randn(2, 20)
        weights = torch.randn(10, 20)
        sim = models.cosine_similarity(feats, weights)

        self.assertEqual(sim.size(), torch.Size((2, 10)))


class ResNetArcFaceTests(unittest.TestCase):

    def test_forward(self):
        inputs: torch.Tensor = torch.randn((2, 3, 128, 128))
        labels: torch.Tensor = torch.Tensor([0, 1])

        params: Dict[str, Any] = {
            's': 30.0,
            'm': 0.5,
            'easy_margin': False
        }
        net = models.ResNetArcFace('resnet18', 10, **params)
        out = net(inputs, labels)

        self.assertEqual(out.size(), torch.Size((2, 10)))

    def test_forward_without_labels(self):
        inputs: torch.Tensor = torch.randn((2, 3, 128, 128))

        params: Dict[str, Any] = {
            's': 30.0,
            'm': 0.5,
            'easy_margin': False
        }
        net = models.ResNetArcFace('resnet18', 10, **params)
        out = net(inputs)

        self.assertEqual(out.size(), torch.Size((2, 10)))


class ResNetCosFaceTests(unittest.TestCase):

    def test_forward(self):
        inputs: torch.Tensor = torch.randn((2, 3, 128, 128))
        labels: torch.Tensor = torch.Tensor([0, 1])

        params: Dict[str, float] = {
            's': 30.0,
            'm': 0.5,
        }
        net = models.ResNetCosFace('resnet18', 10, **params)
        out = net(inputs, labels)

        self.assertEqual(out.size(), torch.Size((2, 10)))

    def test_forward_without_labels(self):
        inputs: torch.Tensor = torch.randn((2, 3, 128, 128))

        params: Dict[str, float] = {
            's': 30.0,
            'm': 0.5,
        }
        net = models.ResNetCosFace('resnet18', 10, **params)
        out = net(inputs)

        self.assertEqual(out.size(), torch.Size((2, 10)))


class ResNetSphereFaceTests(unittest.TestCase):

    def test_forward(self):
        inputs: torch.Tensor = torch.randn((2, 3, 128, 128))
        labels: torch.Tensor = torch.Tensor([0, 1])

        params: Dict[str, int] = {
            'm': 4,
        }
        net = models.ResNetSphereFace('resnet18', 10, **params)
        out = net(inputs, labels)

        self.assertEqual(out.size(), torch.Size((2, 10)))

    def test_forward_without_labels(self):
        inputs: torch.Tensor = torch.randn((2, 3, 128, 128))

        params: Dict[str, int] = {
            'm': 4,
        }
        net = models.ResNetSphereFace('resnet18', 10, **params)
        out = net(inputs)

        self.assertEqual(out.size(), torch.Size((2, 10)))


class IsMetricModel(unittest.TestCase):

    def test_arcface(self):
        arcface = models.ResNetArcFace('resnet18', 2, False, **{})
        self.assertTrue(models.is_metric_model(arcface))

    def test_cosface(self):
        cosface = models.ResNetCosFace('resnet18', 2, False, **{})
        self.assertTrue(models.is_metric_model(cosface))

    def test_sphereface(self):
        sphereface = models.ResNetSphereFace('resnet18', 2, False, **{})
        self.assertTrue(models.is_metric_model(sphereface))

    def test_false(self):
        linear = nn.Linear(2, 2)
        self.assertFalse(models.is_metric_model(linear))


class FreezeResnetTestsEncoder(unittest.TestCase):

    def test_resnet18(self):
        net = torchvision.models.resnet18(False)
        net = models.freeze_resnet_encoder(net)

        for module in net.children():
            module_name: str = module.__class__.__name__
            for params in module.parameters():
                if module_name in ('AdaptiveAvgPool2d', 'Linear'):
                    self.assertTrue(params.requires_grad)
                else:
                    self.assertFalse(params.requires_grad)


class UnfeezeNetworkTests(unittest.TestCase):

    def test_resnet18(self):
        net = torchvision.models.resnet18(False)
        net = models.freeze_resnet_encoder(net)

        for module in net.children():
            module_name: str = module.__class__.__name__
            for params in module.parameters():
                if module_name in ('AdaptiveAvgPool2d', 'Linear'):
                    self.assertTrue(params.requires_grad)
                else:
                    self.assertFalse(params.requires_grad)

        net = models.unfreeze_network(net)
        for module in net.children():
            for params in module.parameters():
                self.assertTrue(params.requires_grad)


class FreezeMetricResnetEncoderTests(unittest.TestCase):

    def test_resnet_arcface(self):
        net = models.ResNetArcFace('resnet18', 1000, False)
        net = models.freeze_metric_resnet_encoder(net)

        for module in net.resnet.children():
            module_name: str = module.__class__.__name__
            for params in module.parameters():
                if module_name in ('AdaptiveAvgPool2d', 'Linear'):
                    self.assertTrue(params.requires_grad)
                else:
                    self.assertFalse(params.requires_grad)

        for params in net.metric_fc.parameters():
            self.assertTrue(params.requires_grad)
