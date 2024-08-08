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
# * Author: Beatrice Alessandra Motetti <beatrice.motetti@polito.it>           *
# *----------------------------------------------------------------------------*

import argparse
import copy
import logging
import math
import os
import pathlib
from datetime import datetime
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from plinio.methods import MixPrec
from plinio.methods.mixprec.nn import MixPrecType
from plinio.methods.mixprec.nn.mixprec_identity import MixPrec_Identity
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrec_Qtz_Layer
from plinio.methods.mixprec.quant.quantizers import PACT_Act_Signed
from pytorch_benchmarks import tiny_imagenet as tin
from pytorch_benchmarks.utils import seed_all
from torch.distributed import ReduceOp, all_reduce, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import swa_utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, resnet18,
                                resnet50)

import imagenet_benchmark as imn
from hardware_models.hardware_model_mpic import (compute_cycles_mpic,
                                                 compute_energy_mpic)
from hardware_models.hardware_model_ne16 import compute_cycles_ne16
from imagenet_benchmark.utils import DDPCheckPoint, ddp_setup, get_free_port


def anneal_temperature(temperature, max_epochs=100):
    # FbNetV2-like annealing
    # return temperature * math.exp(math.log(1e-3)/max_epochs)
    return temperature * math.exp(math.log(math.exp(-0.045 * 500))/max_epochs)


def train_loop(model, epochs, checkpoint_dir, train_dl,
               val_dl, test_dl, device,
               use_ema=None,
               start_epoch=-1,
               optimizer_state_dict=None,
               train_again=False,
               reg_strength=0.0,
               optimizer: Optional[torch.optim.Optimizer]=None,
               arch_optimizer: Optional[torch.optim.Optimizer]=None,
               temperature: float = 1.0,
               search: bool = False,
               source: str = "huggingface"):

    criterion = imn.get_default_criterion()
    earlystop_flag = torch.zeros(1).to(device)
    best_acc1 = 0
    checkpoint = DDPCheckPoint(checkpoint_dir / f"search_{search}_checkpoints",
                               model,
                               optimizer,
                               'max',
                               save_best_only=True,
                               save_last_epoch=True)
    temperature = temperature
    # Train
    if use_ema:
        # `use_buffers=True` ensures update of bn statistics.
        # torch doc says that it may increase accuracy.
        ema_model = swa_utils.AveragedModel(model,
                                            multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.9995),
                                            use_buffers=True)
        checkpoint_ema = DDPCheckPoint(checkpoint_dir / 'ema', ema_model, optimizer, 'max',
                                       save_best_only=True, save_last_epoch=True)
    else:
        ema_model = None

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if arch_optimizer is not None:
        arch_scheduler = StepLR(arch_optimizer, step_size=30, gamma=0.1)
    else:
        arch_scheduler = None

    for epoch in range(start_epoch+1, epochs):
        logging.info(f"Epoch: {epoch}, temperature: {temperature}")
        metrics = imn.train_one_epoch(
            epoch=epoch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train=train_dl,
            val=val_dl,
            device=device,
            ema_model=ema_model,
            source=source,
            search=search,
            reg_strength=reg_strength,
            arch_optimizer=arch_optimizer)

        temperature = anneal_temperature(temperature)
        model.module.update_softmax_temperature(temperature)
        scheduler.step()
        if arch_scheduler is not None:
            arch_scheduler.step()

        # evaluate on validation set
        if val_dl is not None:
            logging.info(f"Val Set Loss: {metrics['val_loss']}")
            logging.info(f"Val Set Accuracy: {metrics['val_acc']}")

        # evaluate on test set
        test_metrics = imn.evaluate(model, criterion, test_dl, device, source=source)
        logging.info(f"Test Set Loss: {test_metrics['loss']}")
        logging.info(f"Test Set Accuracy: {test_metrics['acc']}")

        # EMA
        if ema_model is not None and device == 0:
            ema_test_metrics = imn.evaluate(ema_model, criterion, test_dl, device, source=source)
            logging.info(f"EMA Test Set Loss: {ema_test_metrics['loss']}")
            logging.info(f"EMA Test Set Accuracy: {ema_test_metrics['acc']}")

        acc1 = metrics['val_acc']
        if val_dl is not None:
            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch
                best_acc1 = acc1
                epoch_wout_improve = 0
                logging.info(f'New best Acc_val: {best_acc1}')
            else:
                epoch_wout_improve += 1
                logging.info(f'Epoch without improvement: {epoch_wout_improve}')

        if device == 0:
            if val_dl is not None:
                checkpoint(epoch, metrics['val_acc'])
            else:
                checkpoint(epoch, test_metrics['acc'])
            if use_ema:
                checkpoint_ema(epoch, ema_test_metrics['acc'])


        # Early-Stop
        if epoch_wout_improve >= 20 and device == 0:
            earlystop_flag += 1
        all_reduce(earlystop_flag, op=ReduceOp.SUM)
        if earlystop_flag > 0:  # on all devices
            logging.info(f"GPU {device}, early stopping at epoch: {epoch}, best epoch: {best_epoch}")
            break

    if device == 0:
        # Reload the best model on the main process
        checkpoint.load_best()
        if val_dl is not None:
            val_metrics = imn.evaluate(model, criterion, val_dl, device, source=source)
            test_metrics = imn.evaluate(model, criterion, test_dl, device, source=source)
            logging.info(f"Best Val Set Loss: {val_metrics['loss']}")
            logging.info(f"Best Val Set Accuracy: {val_metrics['acc']}")
            logging.info(f"Test Set Loss @ Best on Val: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy @ Best on Val: {test_metrics['acc']}")
        else:
            test_metrics = imn.evaluate(model, criterion, test_dl, device, source=source)
            logging.info(f"Test Set Loss: {test_metrics['loss']}")
            logging.info(f"Test Set Accuracy: {test_metrics['acc']}")
        if use_ema:
            checkpoint_ema.load_best()
            ema_test_metrics = imn.evaluate(ema_model, criterion, test_dl, device, source=source)
            logging.info(f"EMA Test Set Loss: {ema_test_metrics['loss']}")
            logging.info(f"EMA Test Set Accuracy: {ema_test_metrics['acc']}")


def main(rank, world_size, port, args):
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    N_EPOCHS = args.epochs
    USE_EMA = args.use_ema
    VAL_SPLIT = 0.1
    SOURCE = "original"

    # Set up logging in the worker process
    logging.basicConfig(filename=CHECKPOINT_DIR / 'log.txt',
                        level=logging.INFO,
                        format='%(asctime)s [%(process)d] %(message)s')

    # Setup ddp
    ddp_setup(rank, world_size, port)

    # Ensure determinstic execution
    seed_all(seed=args.seed)

    # get the data
    data_dir = DATA_DIR
    datasets = imn.get_data(data_dir=data_dir,
                            val_split=VAL_SPLIT,
                            seed=args.seed,
                            source=SOURCE)
    dataloaders = imn.build_dataloaders(datasets,
                                        seed=args.seed,
                                        sampler_fn=DistributedSampler,
                                        source=SOURCE,
                                        val_batch_size=args.batch_size,
                                        train_batch_size=args.batch_size)

    train_dl, val_dl, test_dl = dataloaders
    input_shape = (3, 224, 224)

    # get the pre-trained model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(rank)

    # Run eval with pretrained model
    criterion = imn.get_default_criterion()
    pretrained_metrics = imn.evaluate(model, criterion, test_dl, rank, source=SOURCE)
    logging.info(f"Pretrained Test Set Accuracy: {pretrained_metrics['acc']}")

    # --------------------------------------------------------------------
    # SEARCH LOOP
    # --------------------------------------------------------------------
    # Convert the model to the MixPrec format to perform the search
    if args.hardware_model == 'mpic':
        MixPrec.get_cost = compute_cycles_mpic
        logging.info("Chosen regularizer: MPIC")
    elif args.hardware_model == 'ne16':
        MixPrec.get_cost = compute_cycles_ne16
        logging.info("Chosen regularizer: NE16")
    else:
        raise ValueError("Choice of the hardware model '{}' not supported!".format(
            args.hardware_model))

    a_prec = tuple([
        int(el) for el in args.a_precisions.strip('()').split(',') if el.strip().isdigit()])
    w_prec = tuple([
        int(el) for el in args.w_precisions.strip('()').split(',') if el.strip().isdigit()])

    if args.search_type == "channel":
        search_type = MixPrecType.PER_CHANNEL
    elif args.search_type == "layer":
        search_type = MixPrecType.PER_LAYER
    else:
        raise ValueError("Search-type '{}' not supported".format(args.search_type))

    DEFAULT_QINFO_INPUT_QUANTIZER = {
    'a_quantizer': {
        'quantizer': PACT_Act_Signed,
        'kwargs': {
            'init_clip_val_inf': -3,
            'init_clip_val_sup': 3,
            },
        }
    }
    model.eval()
    mixprec_model = MixPrec(model,
                        input_shape=input_shape,
                        activation_precisions=a_prec,
                        weight_precisions=w_prec,
                        w_mixprec_type=search_type,
                        temperature=1,
                        gumbel_softmax=False,
                        hard_softmax=False,
                        qinfo_input_quantizer=DEFAULT_QINFO_INPUT_QUANTIZER,
                        input_activation_precisions=(8, ))

    mixprec_model = mixprec_model.to(rank)

    # group model/architecture parameters. Do not apply weight decay on the clip values
    net_parameters_with_wd = [
        param[1] for param in mixprec_model.named_net_parameters() if "clip_val" not in param[0]]
    net_parameters_without_wd = [
        param[1] for param in mixprec_model.named_net_parameters() if "clip_val" in param[0]]
    nas_parameters = [param[1] for param in mixprec_model.named_nas_parameters()]

    # define both the net and the nas parameters' optimizers
    param_dicts = [
        {'params': net_parameters_without_wd, 'weight_decay': 0},
        {'params': net_parameters_with_wd}]


    optimizer = torch.optim.SGD(param_dicts,
                                lr=args.lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=4e-5)

    arch_optimizer = torch.optim.SGD(nas_parameters,
                                lr=args.lra,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=0)

    # Move model to DDP
    mixprec_model = DDP(mixprec_model, device_ids=[rank], find_unused_parameters=True)

    # Run eval with pretrained model but quantized
    criterion = imn.get_default_criterion(label_smoothing=0.)
    # pretrained_metrics = imn.evaluate(mixprec_model, criterion, test_dl, rank, source=SOURCE)
    # logging.info(f"Pretrained quantized Test Set Accuracy: {pretrained_metrics['acc']}")

    # Search Phase
    train_loop(model=mixprec_model,
               epochs=N_EPOCHS,
               checkpoint_dir=CHECKPOINT_DIR,
               train_dl=train_dl,
               val_dl=val_dl,
               test_dl=test_dl,
               device=rank,
               use_ema=USE_EMA,
               reg_strength=args.reg_strength,
               optimizer=optimizer,
               arch_optimizer=arch_optimizer,
               search=True,
               source=SOURCE)
    logging.info(f"alpha summary {mixprec_model.module.alpha_summary()}")
    logging.info(f"model size: {mixprec_model.module.get_size()}")

    # Evaluation after search
    pretrained_metrics = imn.evaluate(mixprec_model, criterion, test_dl, rank, source=SOURCE)
    logging.info(f"After Search Test Set Accuracy: {pretrained_metrics['acc']}")

    if rank ==0:
        torch.save(mixprec_model.module, CHECKPOINT_DIR / 'final_model_search.pth')
    mixprec_model.eval()
    cycles = mixprec_model.module.get_cost()
    logging.info("model cycles: {}".format(cycles))
    if args.hardware_model == 'mpic':
        logging.info("model energy: {}".format(compute_energy_mpic(cycles)))
    else:
        logging.info("model energy: -1")

    # --------------------------------------------------------------------
    # FINE-TUNING LOOP
    # --------------------------------------------------------------------
    # force hard softmax sampling so that the highest coefficients are taken for the precision
    # selection.Freeze the alpha so that they remain the same ones.
    mixprec_model.module.update_softmax_options(False, True, False)
    mixprec_model.module.freeze_alpha()

    # Freeze the clip val update for the activations quantizers which are not part of the final
    # architecture
    for layer in mixprec_model.module._target_layers:
        if isinstance(layer, MixPrec_Identity):
            continue
        if isinstance(layer.mixprec_a_quantizer, MixPrec_Qtz_Layer):
            index_a_prec = torch.argmax(layer.mixprec_a_quantizer.theta_alpha).item()
            for i in range(len(layer.mixprec_a_quantizer.mix_qtz)):
                if i != index_a_prec:
                    layer.mixprec_a_quantizer.mix_qtz[i].clip_val.requires_grad = False
                    layer.mixprec_a_quantizer.mix_qtz[i].clip_val.grad = None
        if isinstance(layer.input_quantizer, MixPrec_Qtz_Layer):
            if isinstance(layer.input_quantizer.mix_qtz[0], PACT_Act_Signed):
                index_a_prec = torch.argmax(layer.input_quantizer.theta_alpha).item()
                for i in range(len(layer.input_quantizer.mix_qtz)):
                    if i != index_a_prec:
                        layer.input_quantizer.mix_qtz[i].clip_val_inf.requires_grad = False
                        layer.input_quantizer.mix_qtz[i].clip_val_inf.grad = None
                        layer.input_quantizer.mix_qtz[i].clip_val_sup.requires_grad = False
                        layer.input_quantizer.mix_qtz[i].clip_val_sup.grad = None

    exported_model = mixprec_model

    net_parameters_with_wd = [
        param[1] for param in exported_model.module.named_net_parameters() if "clip_val" not in param[0]]
    net_parameters_without_wd = [
        param[1] for param in exported_model.module.named_net_parameters() if "clip_val" in param[0]]

    # define both the net and the nas parameters' optimizers
    param_dicts = [
        {'params': net_parameters_without_wd, 'weight_decay': 0},
        {'params': net_parameters_with_wd}]

    optimizer = torch.optim.SGD(param_dicts,
                                lr=args.lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=4e-5)

    # Run eval before fine-tuning
    criterion = imn.get_default_criterion()
    pretrained_metrics = imn.evaluate(exported_model, criterion, test_dl, rank, source=SOURCE)
    logging.info(f"Before fine-tuning Test Set Accuracy: {pretrained_metrics['acc']}")

    train_loop(model=exported_model,
               epochs=N_EPOCHS,
               checkpoint_dir=CHECKPOINT_DIR,
               train_dl=train_dl,
               val_dl=val_dl,
               test_dl=test_dl,
               device=rank,
               use_ema=USE_EMA,
               reg_strength=args.reg_strength,
               optimizer=optimizer,
               search=False,
               source=SOURCE)

    # Evaluation after fine-tuning
    pretrained_metrics = imn.evaluate(exported_model, criterion, test_dl, rank, source=SOURCE)
    logging.info(f"After Fine-tuning Test Set Accuracy: {pretrained_metrics['acc']}")
    if rank ==0:
        torch.save(exported_model.module, CHECKPOINT_DIR / 'final_model_finetuned.pth')
    exported_model.eval()
    exported_model.module.eval()
    cycles = exported_model.module.get_cost()
    logging.info("model cycles: {}".format(cycles))
    if args.hardware_model == 'mpic':
        logging.info("model energy: {}".format(compute_energy_mpic(cycles)))
    else:
        logging.info("model energy: -1")

    if args.hardware_model == 'mpic':
        logging.info("MPIC model cycles: {}".format(exported_model.module.get_cost().item()))
        MixPrec.get_cost = compute_cycles_ne16
        mixprec_model_ne16 = MixPrec(model,
                        input_shape=input_shape,
                        activation_precisions=a_prec,
                        weight_precisions=w_prec,
                        w_mixprec_type=search_type,
                        temperature=1,
                        gumbel_softmax=False,
                        hard_softmax=False,
                        qinfo_input_quantizer=DEFAULT_QINFO_INPUT_QUANTIZER,
                        input_activation_precisions=(8, ))


        CHECKPOINT = torch.load(args.checkpoint_dir / f"search_False_checkpoints/best.pt", map_location="cpu")
        mixprec_model_ne16.load_state_dict(CHECKPOINT['model_state_dict'])
        mixprec_model_ne16.eval()
        mixprec_model_ne16.update_softmax_options(False, True, False)
        cost = mixprec_model_ne16.get_cost().item()
        logging.info("NE16 model cycles: {}".format(cost))

    else:
        logging.info("NE16 model cycles: {}".format(exported_model.module.get_cost().item()))
        MixPrec.get_cost = compute_cycles_mpic
        mixprec_model_mpic = MixPrec(model,
                        input_shape=input_shape,
                        activation_precisions=a_prec,
                        weight_precisions=w_prec,
                        w_mixprec_type=search_type,
                        temperature=1,
                        gumbel_softmax=False,
                        hard_softmax=False,
                        qinfo_input_quantizer=DEFAULT_QINFO_INPUT_QUANTIZER,
                        input_activation_precisions=(8, ))

        CHECKPOINT = torch.load(args.checkpoint_dir / f"search_False_checkpoints/best.pt", map_location="cpu")
        mixprec_model_mpic.load_state_dict(CHECKPOINT['model_state_dict'])
        mixprec_model_mpic.eval()
        mixprec_model_mpic.update_softmax_options(False, True, False)
        cost = mixprec_model_mpic.get_cost().item()
        logging.info("NE16 model cycles: {}".format(cost))

    logging.info(f"model size: {mixprec_model.module.get_size()}")

    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Training')
    # parser.add_argument('--arch', type=str, help=f'Arch name taken from {model_names}')
    parser.add_argument('--data-dir', type=str, default="./imagenet-1k/imagenet",
                        help='Path to Directory with Training Data')
    parser.add_argument('--checkpoint-dir', type=str, default="experiments_imn_resnet50",
                        help='Path to Directory where to save checkpoints')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp, if not provided will be current time')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs',
                        default=1)
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Dry run with test-set using passed pretrained model')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG Seed, if not provided will be random')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--use-amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='Use Exponential Moving Average')
    parser.add_argument('--hardware-model', type=str,
                        help='Hardware model to use for the regularizer')
    parser.add_argument('--search-type', type=str,
                        help='Type of search to perform, either `layer` or `channel`')
    parser.add_argument('--a-precisions', type=str,
                        help='Activation precisions to consider')
    parser.add_argument('--w-precisions', type=str,
                        help='Weight precisions to consider')
    parser.add_argument('--reg-strength', type=float, default=0.0,
                        help='Regularization strength')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lra', type=float, default=1e-3)

    args = parser.parse_args()

    # Set-up directories
    if args.checkpoint_dir is None:
        args.checkpoint_dir = pathlib.Path().cwd()
    else:
        args.checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.checkpoint_dir = args.checkpoint_dir / f"{args.hardware_model}" / f"aprec{args.a_precisions}_wprec{args.w_precisions}_lr{args.lr}_lra{args.lra}" / f"lambda{args.reg_strength}_bs{args.batch_size}" / args.timestamp
    # Maybe create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging in the main process
    logging.basicConfig(filename=args.checkpoint_dir / 'log.txt',
                        level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')
    logging.info("Process has PID {}".format(os.getpid()))
    logging.info(str(args))

    world_size = args.world_size
    port = get_free_port()
    mp.spawn(main, args=(world_size, port, args), nprocs=world_size)