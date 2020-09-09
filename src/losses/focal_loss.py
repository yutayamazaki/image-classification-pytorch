import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Focal Loss: https://arxiv.org/abs/1708.02002

    Args:
        gamma (float): It modulates the loss for each classes.

        size_average (bool): If True, it returns loss.mean(),
                             else it returns loss.sum().
    """

    def __init__(self, gamma: float = 2.0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma: float = gamma
        self.size_average: bool = size_average

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y_true (torch.Tensor): ground truths with shape (B, ).

            y_pred (torch.Tensor): Predicted label with shape (B, num_classes).

        Returns:
            torch.Tensor: Computed FocalLoss.
        """
        pt = F.log_softmax(y_pred, dim=1).exp()
        loss = -1 * ((1 - pt)**self.gamma * torch.log(pt))

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
