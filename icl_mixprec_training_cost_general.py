import argparse
import copy
import datetime
import logging
import math
import os
import pathlib
from typing import Dict, cast

import pytorch_benchmarks.image_classification as icl
import torch
import torch.nn as nn
from plinio.methods import MixPrec
from plinio.methods.mixprec.nn import MixPrecType
from plinio.methods.mixprec.nn.mixprec_identity import MixPrec_Identity
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrec_Qtz_Layer
from plinio.methods.mixprec.quant.quantizers import PACT_Act_Signed, PACT_Act
from pytorch_benchmarks.utils import (AverageMeter, CheckPoint, EarlyStopping,
                                      accuracy, seed_all)
from pytorch_model_summary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from hardware_models.hardware_model_mpic import (compute_cycles_mpic,
                                                 compute_energy_mpic)
from hardware_models.hardware_model_ne16 import compute_cycles_ne16

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# Definition of the temperature annealing function
def anneal_temperature(temperature):
    # FbNetV2-like annealing
    return temperature * math.exp(-0.045)


def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        reg_strength: float = 0.) -> Dict[str, float]:

    model.eval()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0

    with torch.no_grad():
        for image, target in data:
            step += 1

            image, target = image.to(device), target.to(device)
            output = model(image)
            loss_task = criterion(output, target)

            if search:
                loss_reg = reg_strength * cast(MixPrec, model).get_cost()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0.

            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], image.size(0))
            avgloss.update(loss.detach(), image.size(0))
            avglosstask.update(loss_task.detach(), image.size(0))
            avglossreg.update(loss_reg, image.size(0))

        final_metrics = {
            'loss': avgloss.get(),
            'loss_task': avglosstask.get(),
            'loss_reg': avglossreg.get(),
            'acc': avgacc.get(),
        }
    return final_metrics


def train_one_epoch(
        epoch: int,
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        arch_optimizer,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        device: torch.device,
        reg_strength: float = 0.) -> Dict[str, float]:

    model.train()
    avgacc = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    avglosstask = AverageMeter('2.5f')
    avglossreg = AverageMeter('2.5f')
    step = 0

    with tqdm(total=len(train_dl), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for image, target in train_dl:
            step += 1
            tepoch.update(1)

            image, target = image.to(device), target.to(device)
            output = model(image)
            loss_task = criterion(output, target)

            if search:
                loss_reg = reg_strength * cast(MixPrec, model).get_cost()
                loss = loss_task + loss_reg
                optimizer.zero_grad()
                arch_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                arch_optimizer.step()
            else:
                loss = loss_task
                loss_reg = 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_tr = accuracy(output, target, topk=(1,))
            avgacc.update(acc_tr[0].detach(), image.size(0))
            avgloss.update(loss.detach(), image.size(0))
            avglosstask.update(loss_task.detach(), image.size(0))
            avglossreg.update(loss_reg, image.size(0))
            if step % 100 == 99:
                tepoch.set_postfix({'loss': avgloss,
                                    'loss_task': avglosstask,
                                    'loss_reg': avglossreg,
                                    'acc': avgacc})

        val_metrics = evaluate(search, model, criterion, val_dl, device, reg_strength)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        test_metrics = evaluate(search, model, criterion, test_dl, device, reg_strength)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        final_metrics = {
            'train_loss': avgloss.get(),
            'train_loss_task': avglosstask.get(),
            'train_loss_reg': avglossreg.get(),
            'train_acc': avgacc.get(),
        }
        logging.info(final_metrics)
        logging.info(val_metrics)
        logging.info(test_metrics)

        final_metrics.update(val_metrics)
        tepoch.set_postfix(final_metrics)
        tepoch.close()

        final_metrics.update(test_metrics)
        final_metrics.update({"epoch": epoch})
        return final_metrics


def main(args):
    N_EPOCHS = args.epochs
    LAMBDA = args.strength

    # epoch at which earlystop counter and checkpoint saving starts. If the number of epochs is
    # below a threshold of 20, then -1 is used to ensure checkpoints are saved from the beginning
    EARLYSTOP_START_EPOCH = (20 if N_EPOCHS > 20 else -1)
    temperature = args.temperature

    # (i) storage path for checkpoints & logs (ii) exp. id definition if not provided as argument
    BASE_PATH_EXPS = args.save_dir_base

    if args.exp_folder == "":
        experiment_folder = "lambda{:.2e}_epochs{}_bs{}_gumbel{}_hard{}".format(
                            args.strength,
                            args.epochs,
                            args.batch_size,
                            args.gumbel_softmax,
                            args.hard_softmax)
    else:
        experiment_folder = args.exp_folder
    if args.exp_string == "":
        experiment_string = "{}_{}".format(
            experiment_folder,
            datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        )
    else:
        experiment_string = args.exp_string
    os.makedirs("{}/{}/{}".format(BASE_PATH_EXPS, experiment_folder, experiment_string),
                exist_ok=True)

    logging.basicConfig(
        filename="{}/{}/{}/LOG_{}.log".format(
            BASE_PATH_EXPS, experiment_folder, experiment_string, experiment_string),
        level=logging.DEBUG,
        format='%(message)s')

    logger = logging.getLogger("icl_logger")
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    logger.info(f"EXPERIMENT: <{experiment_string}>")
    logger.info("Process has PID {}".format(os.getpid()))

    logger.info(str(args))

    # check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    seed_all(seed=14)

    # get the data
    datasets = icl.get_data(data_dir=args.data_dir, perf_samples=True)
    dataloaders = icl.build_dataloaders(datasets, batch_size=args.batch_size)
    train_dl, val_dl, test_dl = dataloaders

    datasets_all = icl.get_data(data_dir=args.data_dir, perf_samples=False)
    dataloaders_all = icl.build_dataloaders(datasets_all, args.batch_size)
    _, _, test_dl_all = dataloaders_all

    # get the model
    model = icl.get_reference_model('resnet_8')
    model = model.to(device)
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape
    logger.info(summary(model, input_example, show_input=False, show_hierarchical=True))

    # --------------------------------------------------------------------
    # WARMUP LOOP
    # --------------------------------------------------------------------
    criterion = icl.get_default_criterion()

    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    scheduler = icl.get_default_scheduler(optimizer)

    earlystop = EarlyStopping(patience=args.patience, mode='max')

    if pathlib.Path(f'{BASE_PATH_EXPS}/final_best_warmup.ckp').exists():
        checkpoint = torch.load(f'{BASE_PATH_EXPS}/final_best_warmup.ckp')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Skipping warmup, model loaded from '{BASE_PATH_EXPS}/final_best_warmup.ckp'")
    else:
        logger.info(f"Running warmup,'{BASE_PATH_EXPS}/final_best_warmup.ckp' does not exist")
        warmup_checkpoint = CheckPoint('{}/{}/{}/warmup_checkpoints'.format(
            BASE_PATH_EXPS, experiment_folder, experiment_string), model, optimizer, 'max')

        for epoch in range(N_EPOCHS):
            logger.info("Epoch {}".format(epoch))
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, None, train_dl, val_dl, test_dl, device)
            scheduler.step()

            if epoch > EARLYSTOP_START_EPOCH:
                warmup_checkpoint(epoch, metrics['val_acc'])
                if earlystop(metrics['val_acc']):
                    logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                    break

            if epoch == (N_EPOCHS - 1):
                logger.info("Stopped at epoch {} because of maximum epochs limit".format(epoch))

        warmup_checkpoint.load_best()
        warmup_checkpoint.save(f'{BASE_PATH_EXPS}/final_best_warmup.ckp')

    # perform evaluation, either if the model has been loaded from a previous checkpoint
    # or has been trained
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    test_metrics_all = evaluate(False, model, criterion, test_dl_all, device)
    logger.info("Warmup Test Set Loss: {} - entire test set loss: {}".format(
        test_metrics['loss'], test_metrics_all['loss']))
    logger.info("Warmup Test Set Accuracy: {} - entire test set accuracy: {}".format(
        test_metrics['acc'], test_metrics_all['acc']))

    # --------------------------------------------------------------------
    # SEARCH LOOP
    # --------------------------------------------------------------------
    # Convert the model to the MixPrec format to perform the search
    if args.hardware_model == 'mpic':
        MixPrec.get_cost = compute_cycles_mpic
        logger.info("Chosen regularizer: MPIC")
    elif args.hardware_model == 'ne16':
        MixPrec.get_cost = compute_cycles_ne16
        logger.info("Chosen regularizer: NE16")
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

    mixprec_model = MixPrec(copy.deepcopy(model),
                            input_shape=input_shape,
                            activation_precisions=a_prec,
                            weight_precisions=w_prec,
                            w_mixprec_type=search_type,
                            temperature=temperature,
                            gumbel_softmax=args.gumbel_softmax,
                            hard_softmax=args.hard_softmax)

    mixprec_model = mixprec_model.to(device)
    logger.info(f"alpha summary {mixprec_model.alpha_summary()}")

    # group model/architecture parameters. Do not apply weight decay on the clip values
    net_parameters_with_wd = [
        param[1] for param in mixprec_model.named_net_parameters() if "clip_val" not in param[0]]
    net_parameters_without_wd = [
        param[1] for param in mixprec_model.named_net_parameters() if "clip_val" in param[0]]
    nas_parameters = [param[1] for param in mixprec_model.named_nas_parameters()]

    criterion = icl.get_default_criterion()

    # define both the net and the nas parameters' optimizers
    param_dicts = [
        {'params': net_parameters_without_wd, 'weight_decay': 0},
        {'params': net_parameters_with_wd}]
    optimizer = torch.optim.Adam(param_dicts,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = icl.get_default_scheduler(optimizer)

    arch_optimizer = torch.optim.SGD(nas_parameters,
                                     lr=args.lra,
                                     momentum=args.momentum,
                                     weight_decay=args.alpha_decay)
    arch_scheduler = icl.get_default_scheduler(arch_optimizer)

    # compute the incremental factor for the regularization strength. The variable scheme,
    # if selected, is anyhow applied only if there is more than one epoch, otherwise there
    # is no need and can cause problems since there is a division by 0.
    if (args.incremental_strength) and (N_EPOCHS > 1):
        increment_lambda = (LAMBDA * 99 / 100) / int(N_EPOCHS / 2)
    else:
        increment_lambda = 0.0
    reg_strength = LAMBDA

    # set EarlyStop with patience and CheckPoint
    search_checkpoint = CheckPoint('{}/{}/{}/search_checkpoints'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string), mixprec_model, optimizer, 'max')
    earlystop = EarlyStopping(patience=args.patience, mode='max')

    for epoch in range(N_EPOCHS):
        if args.incremental_strength:
            reg_strength = min(LAMBDA / 100 + increment_lambda * epoch, LAMBDA)

        logger.info("Epoch {}, temperature = {}, lambda = {}".format(
            epoch, temperature, reg_strength))
        metrics = train_one_epoch(
            epoch, True, mixprec_model, criterion, optimizer, arch_optimizer,
            train_dl, val_dl, test_dl, device, reg_strength=reg_strength)

        logger.info("architectural summary:")
        logger.info(mixprec_model)
        logger.info(f"model size: {mixprec_model.get_size()}")
        logger.info("model cycles: {}".format(mixprec_model.get_cost()))

        if epoch > EARLYSTOP_START_EPOCH:
            search_checkpoint(epoch, metrics['val_acc'])
            if earlystop(metrics['val_acc']):
                logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                break

            if epoch % 25 == 0:
                logger.info("alpha summary {}".format(mixprec_model.alpha_summary()))

        scheduler.step()
        arch_scheduler.step()
        temperature = anneal_temperature(temperature)
        mixprec_model.update_softmax_temperature(temperature)

        if epoch == (N_EPOCHS - 1):
            logger.info("Stopped at epoch {} because of maximum epochs limit".format(epoch))

    # load and evaluate the best model found during the search phase
    logger.info("Load best model from '{}'".format(search_checkpoint.best_path))
    search_checkpoint.load_best()
    search_checkpoint.save('{}/{}/{}/final_best_search.ckp'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string))

    logger.info("final architectural summary:")
    logger.info(mixprec_model)
    logger.info(f"model size: {mixprec_model.get_size()}")
    logger.info("model cycles: {}".format(mixprec_model.get_cost()))

    test_metrics = evaluate(True, mixprec_model, criterion, test_dl, device)
    test_metrics_all = evaluate(True, mixprec_model, criterion, test_dl_all, device)
    logger.info("Search Test Set Loss: {} - entire test set loss: {}".format(
        test_metrics['loss'], test_metrics_all['loss']))
    logger.info("Search Test Set Accuracy: {} - entire test set accuracy: {}".format(
        test_metrics['acc'], test_metrics_all['acc']))

    logger.info(f"alpha summary {mixprec_model.alpha_summary()}")
    
    # --------------------------------------------------------------------
    # FINE-TUNING LOOP
    # --------------------------------------------------------------------
    # force hard softmax sampling so that the highest coefficients are taken for the precision
    # selection.Freeze the alpha so that they remain the same ones.
    mixprec_model.update_softmax_options(False, True, False)
    mixprec_model.freeze_alpha()

    # Freeze the clip val update for the activations quantizers which are not part of the final
    # architecture
    for layer in mixprec_model._target_layers:
        if isinstance(layer, MixPrec_Identity):
            continue
        if isinstance(layer.mixprec_a_quantizer, MixPrec_Qtz_Layer):
            index_a_prec = torch.argmax(layer.mixprec_a_quantizer.theta_alpha).item()
            for i in range(len(layer.mixprec_a_quantizer.mix_qtz)):
                if i != index_a_prec:
                    layer.mixprec_a_quantizer.mix_qtz[i].clip_val.requires_grad = False
                    layer.mixprec_a_quantizer.mix_qtz[i].clip_val.grad = None

    exported_model = mixprec_model

    criterion = icl.get_default_criterion()

    net_parameters_with_wd = [
        param[1] for param in exported_model.named_net_parameters() if "clip_val" not in param[0]]
    net_parameters_without_wd = [
        param[1] for param in exported_model.named_net_parameters() if "clip_val" in param[0]]

    param_dicts = [
        {'params': net_parameters_without_wd, 'weight_decay': 0},
        {'params': net_parameters_with_wd}]
    optimizer = torch.optim.Adam(param_dicts,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = icl.get_default_scheduler(optimizer)

    finetuning_checkpoint = CheckPoint('{}/{}/{}/finetuning_checkpoints'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string), exported_model, optimizer, 'max')
    earlystop = EarlyStopping(patience=args.patience, mode='max')

    for epoch in range(N_EPOCHS):
        logger.info("Epoch {}".format(epoch))
        ft_metrics = train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, None,
            train_dl, val_dl, test_dl, device)

        if epoch > EARLYSTOP_START_EPOCH:
            finetuning_checkpoint(epoch, ft_metrics['val_acc'])
            if earlystop(ft_metrics['val_acc']):
                logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                break

        scheduler.step()

        if epoch == (N_EPOCHS - 1):
            logger.info("Stopped at epoch {} because of maximum epochs limit".format(epoch))

    logger.info(f"alpha summary {mixprec_model.alpha_summary()}")

    # load and evaluate the best model found during the fine-tuning phase
    logger.info("Load best model from '{}'".format(finetuning_checkpoint.best_path))
    finetuning_checkpoint.load_best()
    finetuning_checkpoint.save('{}/{}/{}/final_best_finetuning.ckp'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string))
    test_metrics = evaluate(False, exported_model, criterion, test_dl, device)
    test_metrics_all = evaluate(False, exported_model, criterion, test_dl_all, device)
    logger.info("Fine-tuning Test Set Loss: {} - entire test set loss: {}".format(
        test_metrics['loss'], test_metrics_all['loss']))
    logger.info("Fine-tuning Test Set Accuracy: {} - entire test set accuracy: {}".format(
        test_metrics['acc'], test_metrics_all['acc']))

    model.eval()
    cycles = mixprec_model.get_cost()
    logger.info("model cycles: {}".format(cycles))
    if args.hardware_model == 'mpic':
        logger.info("model energy: {}".format(compute_energy_mpic(cycles)))
    else:
        logger.info("model energy: -1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--strength', type=float, help='Regularization Strength')
    parser.add_argument('--incremental_strength', action='store_true',
                        help='Apply increasing strength training scheme')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs',
                        default=500)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--temperature', type=float,
                        help='Initial temperature value', default=1.)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lra', '--learning-rate-alpha', default=1e-2, type=float,
                        metavar='LR', help='initial alpha learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--ad', '--alpha-decay', default=0., type=float,
                        help='alpha decay (default: 0.)',
                        dest='alpha_decay')
    parser.add_argument('--gumbel-softmax', action='store_true', default=False,
                        dest='gumbel_softmax',
                        help='Whether to use the Gumbel-softmax instead of the softmax')
    parser.add_argument('--hard-softmax', action='store_true', default=False,
                        dest='hard_softmax',
                        help='Whether to use the hard version of the Gumbel-softmax')
    parser.add_argument('--exp-string', default="")
    parser.add_argument('--exp-folder', default="")
    parser.add_argument('--pruning_loss', action='store_true',
                        help='Whether to add a pruning loss to the training loss function')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to be used for training')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience during training')
    parser.add_argument('--w_precisions', type=str,
                        help='A string which can be parsed as tuple containing \
                        the weight precisions to be considered during the search phase')
    parser.add_argument('--a_precisions', type=str,
                        help='A string which can be parsed as tuple containing \
                        the activations precisions to be considered during the search phase')
    parser.add_argument('--hardware-model', type=str, default=None,
                        help='Hardware model to be used as regularizer.\
                             Currently supported: "mpic", "ne16"')
    parser.add_argument('--save-dir-base', type=str, help='Base path to save the experiments\
                        results')
    parser.add_argument('--search-type', type=str, default='channel',
                        help='Type of the search. Currently supported: "channel", "layer"')
    args = parser.parse_args()
    main(args)
