from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision

import layers


def _load_resnet(name: str, pretrained: bool = False) -> nn.Module:
    if name == 'resnet18':
        return torchvision.models.resnet18(pretrained)
    elif name == 'resnet34':
        return torchvision.models.resnet34(pretrained)
    elif name == 'resnet50':
        return torchvision.models.resnet50(pretrained)
    elif name == 'resnet101':
        return torchvision.models.resnet101(pretrained)
    elif name == 'resnet152':
        return torchvision.models.resnet152(pretrained)
    else:
        attributes: Tuple[str, ...] = (
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        )
        raise ValueError(f'name must be in {attributes}.')


def cosine_similarity(
    features: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Cosine similarity used to predict metric learning models.
    Args:
        features (torch.Tensor): With shape (B, feat_dim).
        weights (torch.Tensor): (num_classes, feat_dim).
    Returns:
        torch.Tensor: (B, num_classes).
    """
    f_norm = torch.norm(features, dim=1, keepdim=True)
    w_norm = torch.norm(weights, dim=1, keepdim=True)
    features_normed = features / (f_norm + eps)  # (B. feat_dim)
    weights_normed = weights / (w_norm + eps)  # (num_classes, feat_dim)
    return torch.mm(features_normed, weights_normed.T)


class ResNetArcFace(nn.Module):
    """ResNet with ArcFace layer.
    Args:
        backbone (str): Speficy resnet backbone name.
        out_features (int): Output shape, it means num_classes.
        pretrained (bool): Use ImageNet pretrained bakcbone or not.
    """
    def __init__(
        self,
        backbone: str,
        out_features: int,
        pretrained: bool = False,
        **kwargs
    ):
        super(ResNetArcFace, self).__init__()
        self.resnet = _load_resnet(backbone, pretrained)
        self.metric_fc = layers.ArcMarginProduct(1000, out_features, **kwargs)

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Shape (B, 3, H, W).
            labels (Optional[torch.Tensor]): Shape (B, ).
        """
        features = self.resnet(inputs)
        if labels is None:
            return cosine_similarity(features, self.metric_fc.weight)
        return self.metric_fc(features, labels)


class ResNetCosFace(nn.Module):
    """ResNet with CosFace layer.
    Args:
        backbone (str): Speficy resnet backbone name.
        out_features (int): Output shape, it means num_classes.
        pretrained (bool): Use ImageNet pretrained bakcbone or not.
    """
    def __init__(
        self,
        backbone: str,
        out_features: int,
        pretrained: bool = False,
        **kwargs
    ):
        super(ResNetCosFace, self).__init__()
        self.resnet = _load_resnet(backbone, pretrained)
        self.metric_fc = layers.AddMarginProduct(1000, out_features, **kwargs)

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Shape (B, 3, H, W).
            labels (Optional[torch.Tensor]): Shape (B, ).
        """
        features = self.resnet(inputs)
        if labels is None:
            return cosine_similarity(features, self.metric_fc.weight)
        return self.metric_fc(features, labels)


class ResNetSphereFace(nn.Module):
    """ResNet with SphereFace layer.
    Args:
        backbone (str): Speficy resnet backbone name.
        out_features (int): Output shape, it means num_classes.
        pretrained (bool): Use ImageNet pretrained bakcbone or not.
    """
    def __init__(
        self,
        backbone: str,
        out_features: int,
        pretrained: bool = False,
        **kwargs
    ):
        super(ResNetSphereFace, self).__init__()
        self.resnet = _load_resnet(backbone, pretrained)
        self.metric_fc = layers.SphereProduct(1000, out_features, **kwargs)

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Shape (B, 3, H, W).
            labels (Optional[torch.Tensor]): Shape (B, ).
        """
        features = self.resnet(inputs)
        if labels is None:
            return cosine_similarity(features, self.metric_fc.weight)
        return self.metric_fc(features, labels)


def is_metric_model(model: nn.Module) -> bool:
    metric_models: Tuple[str, ...] = (
        'resnetarcface', 'resnetcosface', 'resnetsphereface'
    )
    class_name: str = model.__class__.__name__.lower()
    return class_name in metric_models


def freeze_resnet_encoder(
    resnet: torchvision.models.ResNet
) -> torchvision.models.ResNet:
    """Freeze the parameters of ResNet.
    Args:
        resnet (torchvision.models.ResNet): ResNet to freeze weights.
    Returns:
        torchvision.models.ResNet: Freezed given ResNet.
    """
    for idx, module in enumerate(resnet.children()):
        if idx > 7:
            break
        for params in module.parameters():
            params.requires_grad = False
    return resnet


def unfreeze_network(
    net: nn.Module
) -> torchvision.models.ResNet:
    """Unfreeze the parameters of nn.Module.
    Args:
        net (nn.Module): Networks to unfreeze weights.
    Returns:
        nn.Module: Unfreezed given network.
    """
    for module in net.children():
        for params in module.parameters():
            params.requires_grad = True
    return net


def freeze_metric_resnet_encoder(resface: nn.Module) -> nn.Module:
    """Freeze the parameters of ResNetArcFace, ResNetCosFace, ResNetSphereFace
    Args:
        resface (ResNetArcFace | ResNetCosFace | ResNetSphereFace): Model.
    Returns:
        ResNetArcFace | ResNetCosFace | ResNetSphereFace: Freezed model.
    """
    resface.resnet = freeze_resnet_encoder(resface.resnet)
    return resface
