import torch


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Implementation of accuracy score for torch.Tensor.

    Prameters
    ---------
    y_true: torch.Tensor
        Truth label with 1D.

    y_pred: torch.Tensor
        Predicted label with 1D or 2D(batch, class, ).

    Returns
    -------
    accuracy: float
        Accuracy score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f'y_true and y_pred are inconsistent length, must be'
                         f' same length. Got length y_true:{len(y_true)} and'
                         f'y_pred: {len(y_pred)})')

    if y_pred.dim() == 2:
        _, y_pred = y_pred.max(dim=1)
        y_pred = y_pred.float()
    elif y_pred.dim() >= 3:
        raise ValueError(f'The shape of y_pred must be (batch, class, ) or '
                         f'(batch, ). But got shape {y_pred.size()}')

    acc = torch.eq(y_true, y_pred).float().mean().item()
    return acc
