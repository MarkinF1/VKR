import torch as t
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data): # возвращает указатель на первый элемент
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0] # превращает его в строку и возвращает первый элемент
    if isinstance(data, t.Tensor):
        return data.item()