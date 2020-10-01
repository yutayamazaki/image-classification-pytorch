import argparse
import os
import time
from typing import Any, Dict, List

import albumentations as albu
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import cfg_tools
import csv_tools
import datasets
import metrics
import utils
import visualize as vis


def _get_experiment_dir(weights_path: str) -> str:
    """Get experiment directory from weights path."""
    return '/'.join(weights_path.split('/')[:3])


def _get_experiment_datetime() -> str:
    return args.weights_path.split('/')[2]


if __name__ == '__main__':
    sns.set()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights_path', type=str
    )
    parser.add_argument(
        '-c', '--config_path', type=str
    )
    args = parser.parse_args()

    cfg_dict: Dict[str, Any] = utils.load_yaml(args.config_path)
    cfg: utils.DotDict = utils.DotDict(cfg_dict)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, _, X_test, _, _, y_test = \
        datasets.load_paths_and_labels(root='../images')
    testset = datasets.ClassificationDataset(
        X_test,
        y_test,
        albu.core.serialization.from_dict(cfg.albumentations.test)
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    net = cfg_tools.load_model(
        cfg.model.name, cfg.model.num_classes, **cfg.model.params
    )
    net.load_state_dict(torch.load(args.weights_path, map_location=device))
    net = net.to(device)

    y_preds: List[torch.Tensor] = []
    y_trues: List[torch.Tensor] = []
    time_start: float = time.time()
    for inputs, labels in tqdm(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)

        y_preds.append(outputs.detach().cpu())
        y_trues.append(labels.detach().cpu())

    processing_time: float = time.time() - time_start

    outputs = torch.cat(y_preds, dim=0)  # (N, num_classes)
    targets = torch.cat(y_trues, dim=0)  # (N, )
    acc: float = metrics.accuracy_score(outputs, targets)

    print(f'Accuracy: {acc}')

    conf_mat: np.ndarray = confusion_matrix(
        outputs.argmax(dim=1).numpy(),
        targets.numpy(),
        labels=list(range(cfg.num_classes))
    )

    csv_data: List[str] = [
        _get_experiment_datetime(),
        cfg.readme,
        str(acc),
        str(processing_time)
    ]
    csv_tools.append_row('results.csv', csv_data)

    exp_dir: str = _get_experiment_dir(args.weights_path)
    save_path: str = os.path.join(exp_dir, 'confusion_matrix.png')
    vis.plot_confusion_matrix(
        conf_mat,
        save_path,
        normalize=False
    )
