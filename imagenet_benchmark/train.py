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
# * Author:  Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*
import logging
import math
import os
from time import perf_counter
from typing import Dict, Literal, Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_benchmarks.utils import AverageMeter, accuracy
# from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def anneal_temperature(temperature, max_epochs=120):
    # FbNetV2-like annealing
    # return temperature * math.exp(math.log(1e-3)/max_epochs)
    return temperature * math.exp(math.log(math.exp(-0.045 * 500))/max_epochs)

def get_default_criterion(label_smoothing: float = 0.1) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    # Filter parameters that do not require weight decay, namely the biases and
    # BatchNorm weights

    def filter_fn(x):
        if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return x.requires_grad
        elif isinstance(x, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return x.requires_grad if x.ndim > 1 else False
        else:
            return False

    # Filter
    no_decay_params = filter(filter_fn, net.parameters())
    decay_params = filter(lambda x: not filter_fn(x), net.parameters())

    parameters = [
        {'params': decay_params, 'weight_decay': 4e-5},
        {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    return optim.SGD(parameters, lr=0.05, momentum=0.9, nesterov=True)


# def get_default_scheduler(opt: optim.Optimizer,
#                           warmup_iterations: int = 7500, warmup_init_lr: float = 0.05,
#                           max_lr: float = 0.4, min_lr: float = 2e-4,
#                           max_epochs: int = 300,
#                           verbose=False) -> LRScheduler:
#     # scheduler = CosineAnnealingWarmRestarts(opt, T_0=7500, eta_min=2e-4,
#     #                                         verbose=verbose)
#     scheduler = CosineAnnealingCVNets(opt,
#                                       warmup_iterations=warmup_iterations,
#                                       warmup_init_lr=warmup_init_lr,
#                                       max_lr=max_lr,
#                                       min_lr=min_lr,
#                                       max_epochs=max_epochs,
#                                       verbose=verbose)
#     return scheduler


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device,
        source: Literal['original', 'huggingface'] = 'huggingface',
        ema_model: Optional[nn.Module] = None,
        search: bool = False,
        reg_strength: float = 0.0,
        arch_optimizer: Optional[optim.Optimizer] = None,
        ) -> Dict[str, float]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0

    # DistributedSampler
    train.sampler.set_epoch(epoch=epoch)
    # if VariableBatchSamplerDDP is used uncomment following line
    # train.batch_sampler.set_epoch(epoch=epoch)
    # train.batch_sampler.update_scales(epoch=epoch)

    t0 = perf_counter()
    for batch in train:
        print(f'Epoch {epoch} step {step}', end='\r', flush=True)
        if source == 'huggingface':
            image, target = batch['pixel_values'], batch['label']
        else:
            image, target = batch[0], batch[1]

        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        model = model.to(device)
        output = model(image)
        loss_task = criterion(output, target)

        if search:
            loss_reg = reg_strength * model.module.get_regularization_loss()
            loss = loss_task + loss_reg
            optimizer.zero_grad()
            if arch_optimizer is not None:
                arch_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if arch_optimizer is not None:
                arch_optimizer.step()
        else:
            loss = loss_task
            loss_reg = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ema_model is not None:
            ema_model.update_parameters(model)

        acc_val = accuracy(output, target, topk=(1,))
        avgacc.update(acc_val[0], image.size(0))
        avgloss.update(loss.detach(), image.size(0))
        avglosstask.update(loss_task.detach(), image.size(0))
        if isinstance(loss_reg, torch.Tensor):
            avglossreg.update(loss_reg.detach(), image.size(0))
        else:
            avglossreg.update(loss_reg, image.size(0))
        if (step % 50 == 0) or (step < 50):
            logging.info(f'GPU {device}, Epoch: {epoch}, Step: {step}/{len(train)}, '
                         f'Batch/s: {step / (perf_counter() - t0)}, Train Loss: {avgloss}, Train Acc: {avgacc}, '
                         f'Train Loss task: {avglosstask}, Train Loss reg: {avglossreg}')
        step += 1

    logging.info(f'Epoch {epoch}, Time: {perf_counter() - t0}')
    final_metrics = {
            'train_loss': avgloss.get(),
            'train_loss_task': avglosstask.get(),
            'train_loss_reg': avglossreg.get(),
            'train_acc': avgacc.get(),
        }
    if val is not None:
        val_metrics = evaluate(model, criterion, val, device, search=search, reg_strength=reg_strength, source=source)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        final_metrics.update(val_metrics)
    return final_metrics


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        source: Literal['original', 'huggingface'] = 'huggingface',
        search: bool = False,
        reg_strength: float = 0.0,
        ) -> Dict[str, float]:
    assert source in ['original', 'huggingface'], 'Unknown source'
    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for batch in data:
            if source == 'huggingface':
                image, target = batch['pixel_values'], batch['label']
            else:
                image, target = batch[0], batch[1]

            step += 1
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            model = model.to(device)
            output = model(image)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * model.module.get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0.

            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], image.size(0))
            avgloss.update(loss.detach(), image.size(0))
            avglosstask.update(loss_task.detach(), image.size(0))
            if isinstance(loss_reg, torch.Tensor):
                avglossreg.update(loss_reg.detach(), image.size(0))
            else:
                avglossreg.update(loss_reg, image.size(0))

        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
    return final_metrics