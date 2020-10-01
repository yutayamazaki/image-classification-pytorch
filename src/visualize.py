import itertools
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    save_path: str,
    title: str = 'Confusion Matrix',
    cmap=None,
    normalize: bool = False,
    target_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 9)
) -> None:
    """Plot confusion matrix and save figure.
    Args:
        conf_mat (np.ndarray): Confusion matrix with shape (N, N).
        save_path (str): A path to save figure.
        title (str): Title of the figure.
        cmap: Color maps.
        normalize (bool): Normalize given confusion matrix or not,
                          default is True.
        target_names: (Optional[List[str]]): List of target labels like
                                             ['dog', 'cat', ...]
        figsize (Tuple[int, int]): Size of figure to save.
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        conf_mat = \
            conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)

    if target_names is not None:
        tick_marks: List[int] = list(range(len(target_names)))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh: float = conf_mat.max() / 1.5 if normalize else conf_mat.max() / 2
    for i, j in itertools.product(
        range(conf_mat.shape[0]), range(conf_mat.shape[1])
    ):
        color: str = 'white' if conf_mat[i, j] > thresh else 'black'
        if normalize:
            plt.text(
                j, i, f'{conf_mat[i, j]:.3f}',
                horizontalalignment='center',
                color=color
            )
        else:
            plt.text(
                j, i, f'{conf_mat[i, j]:,}',
                horizontalalignment='center',
                color=color
            )

    plt.tight_layout()
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')

    plt.savefig(save_path)


class FeatureExtractor:

    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model: nn.Module = model
        self.target_layers: List[str] = target_layers
        self.gradients: List[torch.Tensor] = []

    def save_gradient(self, grad: torch.Tensor):
        self.gradients.append(grad)

    def __call__(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (1, 3, H, W).
        """
        outputs: List[torch.Tensor] = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == 'fc':
                continue
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs.append(x)
        return outputs, x


class ModelWrapper:

    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model: nn.Module = model
        self.feature_extractor = FeatureExtractor(model, target_layers)

    def get_gradients(self) -> List[torch.Tensor]:
        return self.feature_extractor.gradients

    def __call__(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (1, B, H, W).
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]:
                All outputs of in each target_layers and output of the model.
        """
        features, outputs = self.feature_extractor(x)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.model.fc(outputs)  # type: ignore
        return features, outputs


class GradCAM:
    """Implementation of GradCAM which visualize what cnn looks.
    Args:
        model (nn.Module): CNN module.
        target_layers (List[str]): A list of layer names used to check grads.
        device (Optional[str]): 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        device: Optional[str] = None
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model: nn.Module = model.to(self.device)
        self.model.eval()
        self.extractor = ModelWrapper(self.model, target_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def __call__(
        self,
        inputs: torch.Tensor,
        index: Optional[int] = None
    ) -> np.ndarray:
        """Forward processing and return heatmap.
        Args:
            inputs (torch.Tensor): (1, C, H, W).
            index (Optional[int]): Specify which class to visualize.
        Returns:
            np.ndarray: Visualized image with shape (H, W).
        """
        inputs = inputs.to(self.device)
        features, outputs = self.extractor(inputs)
        if index is None:
            index = np.argmax(outputs.detach().cpu().numpy())

        one_hot_arr: np.ndarray = np.zeros(
            (1, outputs.size()[-1]), dtype=np.float32
        )
        one_hot_arr[0][index] = 1
        one_hot: torch.Tensor = Variable(
            torch.as_tensor(one_hot_arr), requires_grad=True
        ).to(self.device)

        one_hot = torch.sum(one_hot * outputs)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # (B, C, H, W)
        grads_list: List[torch.Tensor] = self.extractor.get_gradients()
        grads: np.ndarray = grads_list[-1].detach().cpu().numpy()

        # (B, C, H, W) -> (C, H, W)
        targets = features[-1].detach().cpu().numpy()[0, :]

        # (C, )
        weights = np.mean(grads, axis=(2, 3))[0, :]
        # (H, W)
        cam = np.zeros(targets.shape[1:], dtype=np.float32)
        for i, alpha in enumerate(weights):
            cam += alpha * targets[i, :, :]

        cam = np.maximum(cam, 0)
        _, c, h, w = inputs.size()
        cam = cv2.resize(cam, (h, w))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def apply_gradcam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concat original image and mask which generated by GradCAM.
    Args:
        img (np.ndarray): Original image array with shape (H, W, 3).
                          It must be normaized into 0 to 1.
        mask (np.ndarray): 0-1 normalized mask with shape (H, W).
    Returns:
        np.ndarray: Concatnated image with shape (H, W, 3).
    """
    # Check image and mask are normalized.
    assert img.max() <= 1. and img.min() >= 0.
    assert mask.max() <= 1. and mask.min() >= 0.

    heatmap: np.ndarray = cv2.applyColorMap(
        np.uint8(255 * mask), cv2.COLORMAP_JET
    )  # (H, W, 3)
    heatmap = np.float32(heatmap) / 255
    cam: np.ndarray = heatmap + np.float32(img)  # (H, W, 3)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
