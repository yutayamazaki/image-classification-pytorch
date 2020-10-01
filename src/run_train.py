import logging.config
import os
from datetime import datetime
from logging import getLogger
from typing import Any, Callable, Dict, List, Tuple

import albumentations as albu
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

import cfg_tools
import datasets
import metrics
import models
import utils
from models import is_metric_model


def make_experimental_dirs() -> str:
    dirname: str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    save_dir: str = os.path.join('../experiments', dirname)
    os.makedirs(save_dir, exist_ok=False)
    weights_dir: str = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=False)
    return save_dir


def epoch_train(
    net: nn.Module, train_loader, criterion: Callable, optimizer, device: str
) -> Tuple[float, float]:
    train_loss: float = 0.0
    train_acc: float = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if is_metric_model(net):
            outputs = net(inputs, labels)
        else:
            outputs = net(inputs)

        acc: float = metrics.accuracy_score(
            outputs, labels
        )
        train_acc += acc / len(train_loader)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
    return train_loss, train_acc


def epoch_eval(
    net: nn.Module, eval_loader, criterion: Callable, device: str
) -> Tuple[float, float]:
    eval_loss: float = 0.0
    eval_acc: float = 0.0
    for inputs, labels in eval_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_metric_model(net):
            outputs = net(inputs, labels)
        else:
            outputs = net(inputs)

        acc: float = metrics.accuracy_score(
            outputs, labels
        )
        eval_acc += acc / len(eval_loader)

        loss = criterion(outputs, labels)

        eval_loss += loss.item() / len(eval_loader)
    return eval_loss, eval_acc


if __name__ == '__main__':
    sns.set()

    log_config: Dict[str, Any] = utils.load_yaml('logger_conf.yml')
    logging.config.dictConfig(log_config)
    logger = getLogger(__name__)

    cfg_dict: Dict[str, Any] = utils.load_yaml('config.yml')
    cfg: utils.DotDict = utils.DotDict(cfg_dict)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, X_valid, _, y_train, y_valid, _ = \
        datasets.load_paths_and_labels(root='../images')

    trainset = datasets.ClassificationDataset(
        X_train,
        y_train,
        albu.core.serialization.from_dict(cfg.albumentations.train)
    )
    validset = datasets.ClassificationDataset(
        X_valid,
        y_valid,
        albu.core.serialization.from_dict(cfg.albumentations.valid)
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=cfg.batch_size, shuffle=False
    )

    net = cfg_tools.load_model(
        cfg.model.name, cfg.model.num_classes, **cfg.model.params
    )
    net = net.to(device)
    if cfg.freeze_encoder:
        if is_metric_model(net):
            net = models.freeze_metric_resnet_encoder(net)
        else:
            net = models.freeze_resnet_encoder(net)

    criterion = cfg_tools.load_loss(cfg.loss.name, **cfg.loss.params)
    optimizer = cfg_tools.load_optimizer(
        net.parameters(), cfg.optimizer.name, **cfg.optimizer.params
    )
    scheduler = cfg_tools.load_scheduler(
        optimizer, cfg.scheduler.name, **cfg.scheduler.params
    )

    save_dir: str = make_experimental_dirs()
    scores: Dict[str, List[float]] = {
        'train_loss': [],
        'valid_loss': [],
        'train_acc': [],
        'valid_acc': []
    }
    best_acc: float = -1.
    for epoch in range(cfg.num_epochs):
        # Freeze weights only 1~5 epochs.
        if cfg.freeze_encoder and epoch == 5:
            net = models.unfreeze_network(net)

        train_loss, train_acc = epoch_train(
            net, trainloader, criterion, optimizer, device
        )
        valid_loss, valid_acc = epoch_eval(net, validloader, criterion, device)
        scheduler.step()

        if valid_acc > best_acc:
            best_acc = valid_acc
            path: str = os.path.join(
                save_dir,
                'weights',
                f'acc{valid_acc:.5f}_epoch{str(epoch).zfill(3)}.pth'
            )
            torch.save(net.state_dict(), path)

        scores['train_loss'].append(train_loss)
        scores['valid_loss'].append(valid_loss)
        scores['train_acc'].append(train_acc)
        scores['valid_acc'].append(valid_acc)

        logger.info(f'EPOCH: [{epoch + 1}/{cfg.num_epochs}]')
        logger.info(
            f'TRAIN LOSS: {train_loss:.8f}, VALID LOSS: {valid_loss:.8f}'
        )
        logger.info(
            f'TRAIN ACC:  {train_acc:.8f}, VALID ACC:  {valid_acc:.8f}'
        )

    cfg_path: str = os.path.join(save_dir, 'config.yml')
    utils.dump_yaml(cfg_path, cfg.todict())

    # Plot metrics.
    plt.plot(scores['train_loss'], label='train')
    plt.plot(scores['valid_loss'], label='valid')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.clf()

    plt.plot(scores['train_acc'], label='train')
    plt.plot(scores['valid_acc'], label='valid')
    plt.title('Accuracy curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'acc.png'))
