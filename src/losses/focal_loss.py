import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Focal Loss: https://arxiv.org/abs/1708.02002
    
    Parameters
    ----------
    gamma: int, float
        It modulates the loss for each classes.
    
    size_average: bool
        If True, it returns loss.mean(), else it returns loss.sum().
    """

    def __init__(self, gamma=2.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_true: torch.Tensor
            Truth label with 1D.
            ex) torch.Tensor([0, 0, 1])

        y_pred: torch.Tensor
            Predicted label with 2D.
            ex) torch.Tensor([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])

        Returns
        -------
        loss: torch.Tensor
            Calculated FocalLoss.
        """
        if y_true.dim() != 1:
            raise ValueError(f'y_true must be 1D. But it got {y_true.dim()}D')
        if y_pred.dim() != 2:
            raise ValueError(f'y_pred must be 2D. But it got {y_pred.dim()}D')

        pt = F.log_softmax(y_pred, dim=1).exp()
        loss = -1 * ((1 - pt)**self.gamma * torch.log(pt))

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()