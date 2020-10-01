import glob
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import consts


class ClassificationDataset(Dataset):
    """ PyTorch dataset for image classification.
    Args:
        X (List[str]): Paths to train image.
        y (List[int]): Labels of each images.
        transform (Callable): A callable instance of albumentations.
    """
    def __init__(
        self, X: List[str], y: List[int], transforms: Callable
    ):
        X_exists, y_exists = self._check_images_exist(X, y)
        self.X: List[str] = X_exists
        self.y: List[int] = y_exists

        self.transforms: Callable = transforms

    def __len__(self) -> int:
        return len(self.X)

    @staticmethod
    def _check_images_exist(
        X: List[str], y: List[int]
    ) -> Tuple[List[str], List[int]]:
        X_, y_ = [], []
        for x_img, y_label in zip(X, y):
            if os.path.exists(x_img):
                X_.append(x_img)
                y_.append(y_label)
            else:
                print(f'Not found {x_img}.')
        return X_, y_

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pil_img: Image.Image = Image.open(self.X[idx]).convert('RGB')
        label: int = self.y[idx]

        img_arr: np.ndarray = np.array(pil_img)
        aug: Dict[str, np.ndarray] = self.transforms(image=img_arr)
        img: torch.Tensor = torch.as_tensor(
            aug['image'].transpose(2, 0, 1)
        ).float()

        return img, label


def load_paths_and_labels(root: str = '../images'):
    img_paths: List[str] = []
    labels: List[int] = []
    for class_idx, class_name in enumerate(consts.class_names):
        paths: List[str] = glob.glob(os.path.join(root, class_name, '*.jpg'))
        img_paths += paths
        labels += [class_idx for _ in range(len(paths))]

    X_train, X_test, y_train, y_test = train_test_split(
        img_paths, labels, test_size=0.2, random_state=428
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=428
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
