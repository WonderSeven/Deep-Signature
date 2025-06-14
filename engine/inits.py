import os,sys
import pdb

from click.core import batch

sys.path.extend('../')
sys.dont_write_bytecode = True
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from engine.common import create_logger, add_filehandler, format_time, Checkpointer
import datasets
import models


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset(cfg):
    params = {
        'root': cfg.data_root,
        'dataset_arg': cfg.data_reg,
        'traj_nums': cfg.traj_nums,
        'groups': cfg.groups,
        'folds': cfg.folds,
        'val_fold_idx': cfg.seed,
    }
    if cfg.data_name in ['EGFR', 'GPCR']:
        params.update({'atom_type': cfg.atom_type})

    dataset = getattr(datasets, cfg.data_name)(**params)
    return dataset


def get_dataloader(cfg):
    dataset = get_dataset(cfg)
    idx_train, id_val, idx_test = dataset.get_idx_split()

    train_loader = DataLoader(Subset(dataset, idx_train), batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(dataset, id_val), batch_size=cfg.groups, num_workers=cfg.num_workers, shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(dataset, idx_test), batch_size=cfg.folds, num_workers=cfg.num_workers, shuffle=False, drop_last=False)

    return (train_loader, val_loader, test_loader)


def get_algorithm(cfg):
    params = {
        'spatial_in_dim': cfg.input_dim,
        'spatial_out_dim': cfg.hidden_dim,
        'spatial_hidden_dim': cfg.hidden_dim,
        'num_clusters': cfg.num_clusters,
        'temporal_out_dim': cfg.hidden_dim,
    }

    if cfg.algorithm in ['UniversalSignature', 'DeepSignature']:
        params.update({'signature_depth': 2})
    elif cfg.algorithm in ['FrameLSTM', 'Graphormer', 'GraphGraphormer']:
        params.update({'local_mode': cfg.local_mode})

    algorithm = getattr(models, cfg.algorithm)(**params)

    return algorithm


def get_optimizer(cfg, algorithm):
    opt_name = cfg.optimizer.lower()
    if opt_name == 'sgd':
        return optim.SGD(params=algorithm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, nesterov=True)
    elif opt_name == 'adam':
        return optim.Adam(params=algorithm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise Exception("Not support opt : {}".format(opt_name))


def get_loss_func(cfg):
    name = cfg.loss_func.lower()
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'f1':
        return nn.L1Loss()
    elif name == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5))  #
    else:
        raise ValueError('criterion should be cross_entropy')


def get_checkpointer(cfg, algorithm, output_path=None):
    return Checkpointer(output_path, algorithm, cfg.seed)


def get_logger(cfg, output_path=None, name=None):
    logger = create_logger('DynamicPhysics')
    cur_time = format_time()

    if name is None:
        log_name = '{}_{}_{}_seed{}_{}.txt'.format(cfg.mode, cfg.data_reg, cfg.algorithm, cfg.seed, cur_time)
    else:
        log_name = '{}_{}_{}_{}_seed{}_{}.txt'.format(name, cfg.mode, cfg.data_reg, cfg.algorithm, cfg.seed, cur_time)

    log_path = os.path.join(output_path, log_name)

    if cfg.record:
        if os.path.exists(log_path):
            os.remove(log_path)
        # save config
        add_filehandler(logger, log_path)

    return logger



