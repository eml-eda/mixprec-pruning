# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

import os
import pathlib
import socket
from typing import Dict

import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributed import init_process_group
from tqdm import tqdm

from pytorch_benchmarks.utils import AverageMeter, accuracy


# Definition of evaluation function
def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        reg_strength: torch.Tensor = torch.tensor(0.)) -> Dict[str, float]:
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for sample, target in data:
            step += 1
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0.
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], sample.size(0))
            avgloss.update(loss, sample.size(0))
            avglosstask.update(loss_task, sample.size(0))
            avglossreg.update(loss_reg, sample.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
    return final_metrics


# Definition of the function to train for one epoch
def train_one_epoch(
        epoch: int,
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        device: torch.device,
        reg_strength: torch.Tensor = torch.tensor(0.)) -> Dict[str, float]:
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train_dl), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for audio, target in train_dl:
            step += 1
            tepoch.update(1)
            audio, target = audio.to(device), target.to(device)
            output = model(audio)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], audio.size(0))
            avgloss.update(loss, audio.size(0))
            avglosstask.update(loss_task, audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss,
                                    'loss_task': avglosstask,
                                    'loss_reg': avglossreg,
                                    'acc': avgacc})
        val_metrics = evaluate(search, model, criterion, val_dl, device, reg_strength)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        test_metrics = evaluate(search, model, criterion, test_dl, device)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
        final_metrics.update(val_metrics)
        final_metrics.update(test_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()
        print(f'===Epoch {epoch}===')
        print(f'Metrics: {final_metrics}')
        print(f'Train Set Task Loss: {final_metrics["loss_task"]}')
        print(f'Train Set Reg Loss: {final_metrics["loss_reg"]}')
        print(f'Train Set Acc: {final_metrics["acc"]}')
        print(f'Val Set Loss: {final_metrics["val_loss"]}')
        print(f'Val Set Accuracy: {final_metrics["val_acc"]}')
        print(f'Test Set Loss: {final_metrics["test_loss"]}')
        print(f'Test Set Accuracy: {final_metrics["test_acc"]}')
        return final_metrics


class DDPCheckPoint():
    """
    save/load a checkpoint based on a metric
    """
    def __init__(self, dir, net, optimizer,
                 mode='min', fmt='ck_{epoch:03d}.pt',
                 save_best_only=False, save_last_epoch=False):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.dir = pathlib.Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.format = fmt
        self.save_best_only = save_best_only
        self.save_last_epoch = save_last_epoch
        self.net = net
        self.optimizer = optimizer
        self.val = None
        self.epoch = None
        self.best_path = None

    def __call__(self, epoch, val):
        val = float(val)
        if self.val is None:
            self.update_and_save(epoch, val)
        elif self.mode == 'min' and val < self.val:
            self.update_and_save(epoch, val)
        elif self.mode == 'max' and val > self.val:
            self.update_and_save(epoch, val)

        if self.save_last_epoch:
            self.save(self.dir / 'last_epoch.pt', val=val)

    def update_and_save(self, epoch, val):
        self.epoch = epoch
        self.val = val
        self.update_best_path()
        self.save()

    def update_best_path(self):
        if not self.save_best_only:
            self.best_path = self.dir / self.format.format(**self.__dict__)
        else:
            self.best_path = self.dir / 'best.pt'

    def save(self, path=None, val=None):
        if path is None:
            path = self.best_path
        if val is None:
            val = self.val
        torch.save({
                  'epoch': self.epoch,
                  'model_state_dict': self.net.module.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'val': val,
                  }, path)

    def load_best(self):
        if self.best_path is None:
            raise FileNotFoundError("Best path not set!")
        self.load(self.best_path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.epoch = checkpoint['epoch']
        self.net.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.val = checkpoint['val']


def ddp_setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port