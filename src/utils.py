import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(path: str, dic: Dict[str, Any]):
    with open(path, 'w') as f:
        yaml.dump(dic, f)


def count_parameters(net: nn.Module, requires_grad: bool = True) -> int:
    """Count the number of parameters given torch model."""
    if requires_grad:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    return sum(p.numel() for p in net.parameters())


def seed_everything(seed: int = 1234):
    """Set seed for every modules."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


class DotDict(dict):

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                self._parse_nested_dict(arg)

        if kwargs:
            self._parse_nested_dict(kwargs)

    def _parse_nested_dict(self, dic: Dict[Any, Any]):
        for k, v in dic.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)
                continue
            self[k] = v

    def todict(self) -> Dict[Any, Any]:
        """Convert DotDict to default dict."""
        dic: Dict[Any, Any] = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                dic[k] = v.todict()
                continue
            dic[k] = v
        return dic

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]
