import torch


def accuracy_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Args:
        outputs (torch.Tensor): (B, num_classes).
        targets (torch.Tensor): (B, ).
    Returns:
        float: Computed accuracy score.
    """
    assert outputs.dim() == 2 and targets.dim() == 1
    outputs = outputs.argmax(dim=1)  # (B, num_classes) -> (B, )
    acc_tensor = (outputs == targets).float().mean()
    return float(acc_tensor)
