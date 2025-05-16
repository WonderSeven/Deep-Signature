# -*- coding:utf-8 -*-
"""
Created by 'tiexin'
"""
import pdb

import wandb
import logging
import numpy as np
from pathlib import Path

import torch
import engine.inits as inits
import engine.train_tools as tv


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.gpu_ids = str(cfg.gpu_ids)
        self.stage = cfg.mode
        self.output_path = Path(cfg.save_path)
        self.output_path.mkdir(exist_ok=True)
        # seed and stage
        inits.set_seed(cfg.seed)

        self.use_cuda = torch.cuda.is_available()

        # data
        self.train_loader, self.val_loader, self.test_loader = inits.get_dataloader(cfg)
        # components
        self.algorithm = inits.get_algorithm(cfg)
        self.optimizer = inits.get_optimizer(cfg, self.algorithm)
        self.loss_func = inits.get_loss_func(cfg)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # multi gpu
        if len(self.gpu_ids.split(',')) > 1:
            self.algorithm = torch.nn.DataParallel(self.algorithm)
            print('GPUs:', torch.cuda.device_count())
            print('Using CUDA...')
        if self.use_cuda:
            self.algorithm.cuda()
            self.loss_func.cuda()

        # checkpoint
        self.checkpointer = inits.get_checkpointer(cfg, self.algorithm, self.output_path)
        self.epochs = cfg.epochs
        self.start_epoch = 0
        self.test_epoch = cfg.test_epoch

        self.best_val = np.inf

        # self.writer = wandb.init(project="{}-{}".format(cfg.data_name, cfg.save_path.split('/')[-1]))
        self.writer = None
        self.logger = inits.get_logger(cfg, self.output_path)
        self.logger.setLevel(logging.INFO)
        self.logger.info(cfg)
        self.set_training_stage(self.stage)

    def set_training_stage(self, stage):
        stage = stage.strip().lower()
        if stage == 'train':
            self.stage = 2

        elif stage == 'val' or stage == 'test':
            self.stage = 1
            # if self.cfg.record:
            self.checkpointer.load_model(self._get_load_name(self.test_epoch))

    @staticmethod
    def _get_load_name(epoch=-1):
        if epoch == -1:
            model_name = 'best'
        elif epoch == -2:
            model_name = 'last'
        else:
            model_name = str(epoch)
        return model_name

    def _train_net(self, epoch):
        return tv.train(self.cfg, epoch, self.algorithm, self.train_loader, self.optimizer, self.loss_func,
                        use_cuda=self.use_cuda, writer=self.writer)

    def _val_net(self, dataloader):
        return tv.val(self.cfg, self.algorithm, dataloader, self.loss_func, use_cuda=self.use_cuda, writer=self.writer)

    def train(self):
        if self.stage >= 2:
            for epoch_item in range(self.start_epoch, self.epochs):
                self.logger.info('==================================== Epoch %d ====================================' % epoch_item)
                train_total_loss, train_class_loss, train_energy_loss, train_regular_loss, train_acc, train_rec, train_f1 = self._train_net(epoch_item)
                val_total_loss, val_class_loss, val_energy_loss, val_regular_loss, val_acc, val_rec, val_f1 = self._val_net(self.val_loader)
                test_total_loss, test_class_loss, test_energy_loss, test_regular_loss, test_acc, test_rec, test_f1 = self._val_net(self.test_loader)
                if self.writer is not None:
                    self.writer.log({'train_class_loss': train_class_loss.avg,
                                     'train_energy_loss': train_energy_loss.avg,
                                     'train_regular_loss': train_regular_loss.avg})
                    self.writer.log({'train_loss': train_total_loss.avg, 'train_acc': train_acc,
                                     'val_loss': val_total_loss.avg, 'val_acc': val_acc,
                                     'test_loss': test_total_loss.avg, 'test_acc': test_acc})
                if val_total_loss.avg < self.best_val:
                    self.best_val = val_total_loss.avg
                    if self.cfg.record: self.checkpointer.save_model('best', epoch_item)

                self.logger.info('Epoch:{} || Train : {:.4E}, Cla: {:.4E}, Ene: {:.4E}, Reg: {:.4f}, Acc: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                    epoch_item, train_total_loss.avg, train_class_loss.avg, train_energy_loss.avg, train_regular_loss.avg, train_acc, train_rec, train_f1))
                self.logger.info('Epoch:{} || Val : {:.4E}, Cla: {:.4E}, Ene: {:.4E}, Reg: {:.4f}, Acc: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                    epoch_item, val_total_loss.avg, val_class_loss.avg, val_energy_loss.avg, val_regular_loss.avg, val_acc, val_rec, val_f1))
                self.logger.info('Epoch:{} || Test : {:.4E}, Cla: {:.4E}, Ene: {:.4E}, Reg: {:.4f}, Acc: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                    epoch_item, test_total_loss.avg, test_class_loss.avg, test_energy_loss.avg, test_regular_loss.avg, test_acc, test_rec, test_f1))
                if self.cfg.record: self.checkpointer.save_model('last', epoch_item)

            self.logger.info('==================================== Final Test ====================================')
            self.checkpointer.load_model(self._get_load_name(self.test_epoch))
            val_total_loss, val_class_loss, val_energy_loss, val_regular_loss, val_acc, val_rec, val_f1 = self._val_net(self.val_loader)
            test_total_loss, test_class_loss, test_energy_loss, test_regular_loss, test_acc, test_rec, test_f1 = self._val_net(self.test_loader)
            self.logger.info('Val : {:.4E}, Cla: {:.4E}, Ene: {:.4E}, Reg: {:.4f}, Acc: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                val_total_loss.avg, val_class_loss.avg, val_energy_loss.avg, val_regular_loss.avg, val_acc, val_rec, val_f1))
            self.logger.info('Test : {:.4E}, Cla: {:.4E}, Ene: {:.4E}, Reg: {:.4f}, Acc: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                test_total_loss.avg, test_class_loss.avg, test_energy_loss.avg, test_regular_loss.avg, test_acc, test_rec, test_f1))

        elif self.stage == 1:
            self.logger.info('==================================== Final Test ====================================')
            test_total_loss, test_class_loss, test_energy_loss, test_regular_loss, test_acc, test_rec, test_f1 = self._val_net(self.test_loader)
            self.logger.info('Test : {:.4E}, Cla: {:.4E}, Ene: {:.4E}, Reg: {:.4f}, Acc: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                test_total_loss.avg, test_class_loss.avg, test_energy_loss.avg, test_regular_loss.avg, test_acc, test_rec, test_f1))
        else:
            raise ValueError('Stage is wrong!')