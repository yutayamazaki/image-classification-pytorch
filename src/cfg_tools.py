from typing import Tuple

import torch
import torch.nn as nn
import torchvision

import models
import losses


def load_optimizer(params, name: str, **kwargs) -> torch.optim.Optimizer:
    """Load PyTorch optimizer."""
    if name == 'SGD':
        return torch.optim.SGD(
            params,
            **kwargs
        )
    else:
        raise ValueError('name must be "SGD."')


def load_scheduler(
    optimizer: torch.optim.Optimizer, name: str, **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    if name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **kwargs
        )
    elif name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            **kwargs
        )
    else:
        msg: str = \
            'name must be CosineAnnealingLR or CosineAnnealingWarmRestarts.'
        raise ValueError(msg)


def load_loss(name: str, **kwargs) -> nn.Module:
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'FocalLoss':
        return losses.FocalLoss(**kwargs)
    else:
        attributes: Tuple[str] = (
            'CrossEntropyLoss', 'FocalLoss'
        )
        raise ValueError(f'name must be in {attributes}.')


def load_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name == 'resnet18':
        net = torchvision.models.resnet18(**kwargs)
        net.fc = nn.Linear(512, num_classes)
        return net
    elif name == 'resnet34':
        net = torchvision.models.resnet34(**kwargs)
        net.fc = nn.Linear(512, num_classes)
        return net
    elif name == 'resnet50':
        net = torchvision.models.resnet50(**kwargs)
        net.fc = nn.Linear(512 * 4, num_classes)
        return net
    elif name == 'resnet101':
        net = torchvision.models.resnet101(**kwargs)
        net.fc = nn.Linear(512 * 4, num_classes)
        return net
    elif name == 'resnet152':
        net = torchvision.models.resnet152(**kwargs)
        net.fc = nn.Linear(512 * 4, num_classes)
        return net
    elif name == 'resnetarcface':
        return models.ResNetArcFace(**kwargs)
    elif name == 'resnetcosface':
        return models.ResNetCosFace(**kwargs)
    elif name == 'resnetsphereface':
        return models.ResNetSphereFace(**kwargs)
    else:
        attributes: Tuple[str, ...] = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnetarcface', 'resnetcosface', 'resnetsphereface'
        )
        raise ValueError(f'name must be in {attributes}.')
