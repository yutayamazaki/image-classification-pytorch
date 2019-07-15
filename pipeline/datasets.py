import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, img_path_list, labels, transform):
        self.img_path_list = img_path_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)
        X = self.transform(img)

        y = self.labels[idx]
        return X, y


def load_images(base_dir, categories):
    img_path_list = []
    labels = []
    for i, cat in enumerate(categories):
        img_dir = base_dir + cat
        img_path = glob.glob(img_dir + '/*')
        img_path_list += img_path
        labels += [i] * len(img_path)

    return np.array(img_path_list), np.array(labels)
