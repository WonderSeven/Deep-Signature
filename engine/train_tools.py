import pdb
import time
import numpy as np
from tqdm import tqdm

import torch
from engine.common import AverageMeter, ProgressMeter
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def train(cfg, epoch, algorithm, train_loader, optimizer, criterion, use_cuda=True, print_freq=50, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    total_losses = AverageMeter('Total Loss', ':.4e')
    class_losses = AverageMeter('Class Loss', ':.4e')
    energy_losses = AverageMeter('Energy Loss', ':.4e')
    regular_losses = AverageMeter('Regular Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), batch_time, total_losses, class_losses, energy_losses,
                             regular_losses, prefix="Epoch: [{}]".format(epoch))

    all_pred_y, all_target_y = [], []
    algorithm.train()
    end = time.time()
    for idx, data in enumerate(train_loader, start=0):
        if use_cuda:
            data = data.cuda()
        batch_size = data.y.size(0)

        pred_y, e_loss, mc_loss, o_loss = algorithm(data)

        cla_loss = criterion(pred_y, data.y.float())
        # cla_loss = criterion(pred_y, data.y.long())

        e_loss = 0.01 * e_loss # 0.01
        cla_loss = 10. * cla_loss # 10.

        local_loss = e_loss + mc_loss + o_loss + cla_loss
        print('Train idx: [{}/{}], Total: {:.4f}, Cla: {:.4f}, Energy: {:.4f}, Cut: {:.4f}, Ortho: {:.4f}'.format(idx,
                                                                                                   len(train_loader),
                                                                                                   local_loss,
                                                                                                   cla_loss,
                                                                                                   e_loss,
                                                                                                   mc_loss,
                                                                                                   o_loss))
        optimizer.zero_grad()
        local_loss.backward()
        optimizer.step()

        total_losses.update(local_loss.item(), batch_size)
        class_losses.update(cla_loss.item(), batch_size)
        energy_losses.update(e_loss.item(), batch_size)
        regular_losses.update((mc_loss + o_loss).item(), batch_size)
        all_pred_y.append(pred_y.round().detach().cpu().numpy())
        all_target_y.append(data.y.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0 and idx != 0:
            progress.print(idx)

    all_pred_y = np.concatenate(all_pred_y, axis=0).reshape(-1)
    all_target_y = np.concatenate(all_target_y, axis=0).reshape(-1)
    # all_pred_y = (all_pred_y > 0.5).astype(np.int64) # delete sigmoid in classifier
    acc = accuracy_score(all_target_y, all_pred_y) * 100.
    recall = recall_score(all_target_y, all_pred_y, average='binary') * 100.
    f1 = f1_score(all_target_y, all_pred_y, average='binary') * 100.
    progress.print(epoch)

    return total_losses, class_losses, energy_losses, regular_losses, acc, recall, f1


@torch.no_grad()
def val(cfg, algorithm, val_loader, criterion, use_cuda=True, writer=None):
    total_losses = AverageMeter('Total Loss', ':.4e')
    class_losses = AverageMeter('Class Loss', ':.4e')
    energy_losses = AverageMeter('Energy Loss', ':.4e')
    regular_losses = AverageMeter('Regular Loss', ':.4e')

    algorithm.eval()
    all_pred_y, all_target_y = [], []
    for idx, data in enumerate(val_loader):
        # if idx == 1: break
        if use_cuda:
            data = data.cuda()
        batch_size = data.y.size(0)

        pred_y, e_loss, mc_loss, o_loss = algorithm(data)
        cla_loss = criterion(pred_y, data.y.float())
        # cla_loss = criterion(pred_y, data.y.long())

        e_loss = 0.01 * e_loss # 0.01
        cla_loss = 10. * cla_loss # 10.

        local_loss = e_loss + mc_loss + o_loss + cla_loss

        total_losses.update(local_loss.item(), batch_size)
        class_losses.update(cla_loss.item(), batch_size)
        energy_losses.update(e_loss.item(), batch_size)
        regular_losses.update((mc_loss + o_loss).item(), batch_size)
        all_pred_y.append(pred_y.round().detach().cpu().numpy())
        all_target_y.append(data.y.cpu().numpy())

    all_pred_y = np.concatenate(all_pred_y, axis=0).reshape(-1)
    all_target_y = np.concatenate(all_target_y, axis=0).reshape(-1)
    acc = accuracy_score(all_target_y, all_pred_y) * 100.
    recall = recall_score(all_target_y, all_pred_y, average='binary') * 100.
    f1 = f1_score(all_target_y, all_pred_y, average='binary') * 100.

    return total_losses, class_losses, energy_losses, regular_losses, acc, recall, f1
