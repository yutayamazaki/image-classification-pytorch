import unittest
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision

import cfg_tools
import models


class LoadOptimizerTests(unittest.TestCase):

    def test_load_sgd(self):
        net: nn.Module = nn.Linear(3, 2)
        name: str = 'SGD'
        kwargs: Dict[str, Any] = {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0003
        }
        opt = cfg_tools.load_optimizer(
            params=net.parameters(),
            name=name,
            **kwargs
        )
        self.assertIsInstance(opt, torch.optim.SGD)

    def test_raise(self):
        net: nn.Module = nn.Sequential(
            torch.nn.Linear(3, 2)
        )
        name: str = 'InvalidOptimizer'
        kwargs = {}
        with self.assertRaises(ValueError):
            cfg_tools.load_optimizer(
                params=net.parameters(),
                name=name,
                **kwargs
            )


class LoadSchedulerTests(unittest.TestCase):

    def setUp(self):
        net: nn.Module = nn.Linear(3, 2)
        name: str = 'SGD'
        kwargs: Dict[str, Any] = {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0003
        }
        self.optimizer = cfg_tools.load_optimizer(
            params=net.parameters(),
            name=name,
            **kwargs
        )

    def test_load_cosine_annealing_lr(self):
        scheduler = cfg_tools.load_scheduler(
            optimizer=self.optimizer,
            name='CosineAnnealingLR',
            **{'T_max': 10}
        )
        self.assertIsInstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        )

    def test_load_cosine_annealing_warm_restart(self):
        scheduler = cfg_tools.load_scheduler(
            optimizer=self.optimizer,
            name='CosineAnnealingWarmRestarts',
            **{'T_0': 10}
        )
        self.assertIsInstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        )

    def test_raise(self):
        with self.assertRaises(ValueError):
            cfg_tools.load_scheduler(
                optimizer=self.optimizer,
                name='InvalidScheduler',
                **{}
            )


class LoadLossTests(unittest.TestCase):

    def test_cross_entropy(self):
        name: str = 'CrossEntropyLoss'
        params: dict = {}
        criterion = cfg_tools.load_loss(name, **params)
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)

    def test_raise(self):
        name: str = 'InvalidLossName'
        params: dict = {}
        with self.assertRaises(ValueError):
            cfg_tools.load_loss(name, **params)


class LoadModelTests(unittest.TestCase):

    def test_load_resnet18(self):
        name: str = 'resnet18'
        num_classes: int = 2
        kwargs: Dict[str, Any] = {
            'pretrained': False,
        }
        model = cfg_tools.load_model(
            name=name,
            num_classes=num_classes,
            **kwargs
        )
        self.assertIsInstance(model, torchvision.models.ResNet)
        self.assertEqual(model.fc.out_features, num_classes)

    def test_load_resnet34(self):
        name: str = 'resnet34'
        num_classes: int = 3
        kwargs: Dict[str, Any] = {
            'pretrained': False,
        }
        model = cfg_tools.load_model(
            name=name,
            num_classes=num_classes,
            **kwargs
        )
        self.assertIsInstance(model, torchvision.models.ResNet)
        self.assertEqual(model.fc.out_features, num_classes)

    def test_load_resnet50(self):
        name: str = 'resnet50'
        num_classes: int = 3
        kwargs: Dict[str, Any] = {
            'pretrained': False,
        }
        model = cfg_tools.load_model(
            name=name,
            num_classes=num_classes,
            **kwargs
        )
        self.assertIsInstance(model, torchvision.models.ResNet)
        self.assertEqual(model.fc.out_features, num_classes)

    def test_load_resnet101(self):
        name: str = 'resnet101'
        num_classes: int = 3
        kwargs: Dict[str, Any] = {
            'pretrained': False,
        }
        model = cfg_tools.load_model(
            name=name,
            num_classes=num_classes,
            **kwargs
        )
        self.assertIsInstance(model, torchvision.models.ResNet)
        self.assertEqual(model.fc.out_features, num_classes)

    def test_load_resnet152(self):
        name: str = 'resnet152'
        num_classes: int = 3
        kwargs: Dict[str, Any] = {
            'pretrained': False,
        }
        model = cfg_tools.load_model(
            name=name,
            num_classes=num_classes,
            **kwargs
        )
        self.assertIsInstance(model, torchvision.models.ResNet)
        self.assertEqual(model.fc.out_features, num_classes)

    def test_load_arcface(self):
        name: str = 'resnetarcface'
        params: Dict[str, Any] = {
            'pretrained': False,
            'out_features': 2,
            'backbone': 'resnet18',
            's': 30.0
        }
        model = cfg_tools.load_model(name, 2, **params)

        self.assertIsInstance(model, models.ResNetArcFace)

    def test_load_cosface(self):
        name: str = 'resnetcosface'
        params: Dict[str, Any] = {
            'pretrained': False,
            'out_features': 2,
            'backbone': 'resnet18',
            's': 30.0,
            'm': 0.40
        }
        model = cfg_tools.load_model(name, 2, **params)

        self.assertIsInstance(model, models.ResNetCosFace)

    def test_load_sphereface(self):
        name: str = 'resnetsphereface'
        params: Dict[str, Any] = {
            'pretrained': False,
            'out_features': 2,
            'backbone': 'resnet18',
            'm': 4
        }
        model = cfg_tools.load_model(name, 2, **params)

        self.assertIsInstance(model, models.ResNetSphereFace)

    def test_raise(self):
        name: str = 'invalid-model'
        num_classes: int = 2
        kwargs: Dict[str, Any] = {
            'pretrained': True,
        }
        with self.assertRaises(ValueError):
            cfg_tools.load_model(
                name=name,
                num_classes=num_classes,
                **kwargs
            )
