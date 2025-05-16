import pdb
import time
import torch
import pickle
import numpy as np
from pympler import asizeof

def format_time():
    return time.strftime("%Y-%m-%d-%H:%M", time.localtime(time.time()))


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def calc_storage_MB(x):
    if isinstance(x, np.ndarray):
        storage_size_bytes = x.nbytes
    elif isinstance(x, torch.Tensor):
        storage_size_bytes = x.element_size() * x.numel()
    else:
        storage_size_bytes = asizeof.asizeof(x)
    storage_size_mb = storage_size_bytes / (1024 * 1024)
    return storage_size_mb


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def set_no_grad(algorithm, filter_name: list):
    for name, param in algorithm.named_parameters():
        if param.requires_grad and name in filter_name:
            param.requires_grad = False


def set_no_grad_module(algorithm, filter_name: str):
    for name, param in algorithm.named_parameters():
        if param.requires_grad and filter_name in name:
            # print('No grad:{}'.format(name))
            param.requires_grad = False


def generate_init_value(module, value_clamp):
    for param in module.parameters():
        param.data = torch.rand_like(param)*(2*value_clamp) - value_clamp


def count_parameters(model, mode='ind'):
    if mode == 'ind':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'layer':
        return sum(1 for p in model.parameters() if p.requires_grad)
    elif mode == 'row':
        n_mask = 0
        for p in model.parameters():
            if p.dim() == 1:
                n_mask += 1
            else:
                n_mask += p.size(0)
        return n_mask


def get_n_param_layer(net, layers):
    n_param = 0
    for name, p in net.named_parameters():
        if any(f"net.{layer}" in name for layer in layers):
            n_param += p.numel()
    return n_param
