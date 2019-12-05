import math

from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


class CosineAnnealingLRWithWarmRestart(_LRScheduler):
    """ Implementation of STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS.
        https://arxiv.org/pdf/1608.03983.pdf

        Args:
            optimizer (torch.optim.Optimizer): Torch optimizer instance.
            init_T_max (int): Number of epochs in initial iteration
                              (before warm restarts).
            eta_min: (float): Minimum learning rate. Default: 0.
            last_epoch (int): An index of last epoch. Default: -1.
            T_mult (float): Iteration factor in each restart.
    """

    def __init__(
        self,
        optimizer,
        init_T_max,
        eta_min=0.0,
        last_epoch=-1,
        T_mult=1.5
    ):
        self.optimizer = optimizer
        self.T_max_each = init_T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.T_mult = T_mult
        super(CosineAnnealingLRWithWarmRestart, self).__init__(optimizer,
                                                               last_epoch)

    def get_lr(self):
        time_to_restart = self.last_epoch // self.T_max_each > 0 and \
                       self.last_epoch % self.T_max_each == 0
        if time_to_restart:
            self.last_epoch = 0
            self.T_max_each = int(self.T_max_each * self.T_mult)

        return [self.eta_min + (base_lr - self.eta_min) * 0.5 *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max_each))
                for base_lr in self.base_lrs]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    import torchvision

    model = torchvision.models.resnet18(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    lr_scheduler = CosineAnnealingLRWithWarmRestart(optimizer, init_T_max=200)

    lr_list = []
    for i in range(600):
        lr_list.append(lr_scheduler.get_lr()[0])
        lr_scheduler.step()
    plt.plot(lr_list)
    plt.show()