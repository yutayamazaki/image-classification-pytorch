import argparse
from typing import Any, Callable, Dict

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import cfg_tools
import utils
import visualize as vis


def _get_image_id() -> str:
    return args.img_path.split('/')[3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights_path', type=str
    )
    parser.add_argument(
        '-c', '--config_path', type=str
    )
    parser.add_argument(
        '-i', '--img_path', type=str
    )
    args = parser.parse_args()

    cfg_dict: Dict[str, Any] = utils.load_yaml(args.config_path)
    cfg: utils.DotDict = utils.DotDict(cfg_dict)

    pil_img: Image.Image = Image.open(args.img_path).convert('RGB')
    img_arr = np.array(pil_img)  # (H, W, 3)

    transforms: Callable = albu.core.serialization.from_dict(
        cfg.albumentations.test.todict()
    )
    aug: Dict[str, np.ndarray] = transforms(image=img_arr)
    inputs: torch.Tensor = torch.as_tensor(
        aug['image'].transpose(2, 0, 1)
    ).float().unsqueeze(0)

    net = cfg_tools.load_model(
        cfg.model.name,
        cfg.model.num_classes,
        **cfg.model.params
    )
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.load_state_dict(torch.load(args.weights_path, map_location=device))
    net = net.to(device)

    gradcam = vis.GradCAM(net, ['layer4'])
    mask = gradcam(inputs)

    img_arr = cv2.resize(img_arr, (cfg.img_size, cfg.img_size))
    img_arr = img_arr / 255.

    gradcam_img: np.ndarray = vis.apply_gradcam_on_image(img_arr, mask)

    img_id: str = _get_image_id()
    plt.imshow(gradcam_img)
    plt.savefig(f'../{img_id}.png')

    print(net(inputs)[0].detach().cpu().numpy())
