import hashlib

import numpy as np
from PIL import Image


def read_image_as_md5(path: str) -> str:
    """Read image as MD5 hashed format.
    Args:
        path (str): A path to image.
    Returns:
        str: MD5 hashed image.
    """
    with open(path, 'rb') as f:
        img_bytes: bytes = f.read()
        return hashlib.md5(img_bytes).hexdigest()


def average_hash(img: Image.Image) -> str:
    """Convert PIL.Image.Image to hashed string.
    Args:
        img (PIL.Image.Image): An image to apply hashing.
    Returns:
        str: Hashed string.
    """
    resized_gray: Image.Image = img.resize((8, 8)).convert('L')
    avg: float = np.mean(resized_gray)
    bit_img: np.ndarray = np.array(resized_gray) >= avg
    bit_flatten: np.ndarray = bit_img.flatten().astype(int)
    return ''.join(map(str, bit_flatten.tolist()))


def hamming_distance(a: str, b: str) -> int:
    assert len(a) == len(b)
    distance: int = 0
    for a_, b_ in zip(a, b):
        if a_ != b_:
            distance += 1
    return distance
