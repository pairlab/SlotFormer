import numpy as np

import torch
import torchvision.transforms as transforms


def compact(l):
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


def np_stack(arrays, axis=0):
    if not arrays:
        return np.array([])
    return np.stack(arrays, axis=axis)


def np_concat(arrays, axis=0):
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=axis)


def torch_stack(arrays, dim=0):
    if not arrays:
        return torch.tensor([])
    return torch.stack(arrays, dim=dim)


class CLEVRTransforms(object):

    def __init__(self, resolution):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            transforms.Normalize((0.5, ), (0.5, )),  # [-1, 1]
            transforms.Resize(resolution),
        ])

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
