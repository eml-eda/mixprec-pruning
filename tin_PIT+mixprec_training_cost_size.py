import argparse
import copy
import datetime
import logging
import math
import os
import pathlib
from typing import Dict, cast

import pytorch_benchmarks.tiny_imagenet as tin
import torch
import torch.nn as nn
from plinio.methods import MixPrec, PIT
from plinio.methods.mixprec.nn.mixprec_identity import MixPrec_Identity
from plinio.methods.mixprec.nn import MixPrecType
from pytorch_benchmarks.utils import (AverageMeter, CheckPoint, EarlyStopping,
                                      accuracy, seed_all)
from pytorch_model_summary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def anneal_temperature(temperature, max_epochs=50):
    # FbNetV2-like annealing
    # return temperature * math.exp(math.log(1e-3)/max_epochs)
    return temperature * math.exp(math.log(math.exp(-0.045 * 500))/max_epochs)


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
        for audio, target in data:
            step += 1

            audio, target = audio.to(device), target.to(device)

            output = model(audio)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * cast(MixPrec, model).get_regularization_loss()
                loss = loss_task + loss_reg
            else:
                loss = loss_task
                loss_reg = 0.

            acc_val = accuracy(output, target, topk=(1,))
            avgacc.update(acc_val[0], audio.size(0))
            avgloss.update(loss.detach(), audio.size(0))
            avglosstask.update(loss_task.detach(), audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))

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
        for audio, target in train_dl:
            step += 1
            tepoch.update(1)

            audio, target = audio.to(device), target.to(device)

            output = model(audio)
            loss_task = criterion(output, target)
            if search:
                loss_reg = reg_strength * cast(MixPrec, model).get_regularization_loss()
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
            avgacc.update(acc_tr[0].detach(), audio.size(0))
            avgloss.update(loss.detach(), audio.size(0))
            avglosstask.update(loss_task.detach(), audio.size(0))
            avglossreg.update(loss_reg, audio.size(0))
            if step:
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

    logger = logging.getLogger("tin_logger")
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    logger.info(f"EXPERIMENT: <{experiment_string}>")
    logger.info("Process has PID {}".format(os.getpid()))

    logger.info(str(args))

    # check CUDA availability and ensure deterministic execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    seed_all(seed=14)

    # get the data
    datasets = tin.get_data(data_dir=args.data_dir, inp_res=224)
    dataloaders = tin.build_dataloaders(datasets, batch_size=args.batch_size)
    train_dl, val_dl, test_dl = dataloaders

    # get the model
    config = {'pretrained': True, 'std_head': True}
    model = tin.get_reference_model('resnet_18', model_config=config)
    model = model.to(device)
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape
    logger.info(summary(model, input_example, show_input=False, show_hierarchical=True))

    # --------------------------------------------------------------------
    # WARMUP LOOP
    # --------------------------------------------------------------------
    criterion = tin.get_default_criterion()

    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    earlystop = EarlyStopping(patience=args.patience, mode='max')

    if pathlib.Path(f'{BASE_PATH_EXPS}/final_best_warmup_224.ckp').exists():
        checkpoint = torch.load(f'{BASE_PATH_EXPS}/final_best_warmup_224.ckp')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Skipping warmup, model loaded from \
                    '{BASE_PATH_EXPS}/final_best_warmup_224.ckp'")
    else:
        logger.info(f"Running warmup,'{BASE_PATH_EXPS}/final_best_warmup_224.ckp' does not exist")
        warmup_checkpoint = CheckPoint('{}/{}/{}/warmup_checkpoints_224'.format(
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
        warmup_checkpoint.save(f'{BASE_PATH_EXPS}/final_best_warmup_224.ckp')




    # perform evaluation, either if the model has been loaded from a previous checkpoint
    # or has been trained
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    logger.info("224 Warmup Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("224 Warmup Test Set Accuracy: {}".format(test_metrics['acc']))

    # --------------------------------------------------------------------
    # Get the Data with 64pixel resolution as full-imagenet
    datasets = tin.get_data(data_dir=args.data_dir, inp_res=64)
    dataloaders = tin.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    # Now, get Model with reduced ResNet18 head
    config = {'pretrained': True,
              'state_dict': model.state_dict(),  # state-dict of previously trained model
              'std_head': False}
    model = tin.get_reference_model('resnet_18', model_config=config)
    model = model.to(device)

    # Model Summary
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape

    # Get Training Settings
    criterion = tin.get_default_criterion()
    optimizer = tin.get_default_optimizer(model)
    scheduler = tin.get_default_scheduler(optimizer)

    earlystop = EarlyStopping(patience=args.patience, mode='max')

    pit_model = PIT(model, input_shape=input_shape)
    pit_model = pit_model.to(device)
    checkpoint_path_base = args.pit_checkpoint
    CHECKPOINT = torch.load(os.path.join(checkpoint_path_base, "final_best_search.ckp"))
    pit_model.load_state_dict(CHECKPOINT["model_state_dict"])
    exported_model = pit_model.arch_export()
    CHECKPOINT = torch.load(os.path.join(checkpoint_path_base, "final_best_finetuning.ckp"))
    exported_model.load_state_dict(CHECKPOINT["model_state_dict"])
    logger.info("Loading seed model from '{}'".format(checkpoint_path_base))
    model = exported_model.to(device)
    # perform evaluation, either if the model has been loaded from a previous checkpoint
    # or has been trained
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    logger.info("64 Warmup Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("64 Warmup Test Set Accuracy: {}".format(test_metrics['acc']))

    # --------------------------------------------------------------------
    # SEARCH LOOP
    # --------------------------------------------------------------------
    # Convert the model to the MixPrec format to perform the search
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

    criterion = tin.get_default_criterion()

    if pathlib.Path(f'{BASE_PATH_EXPS}/final_best_search.ckp').exists():
        checkpoint = torch.load(f'{BASE_PATH_EXPS}/final_best_search.ckp')
        mixprec_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Skipping search, model loaded from \
                    '{BASE_PATH_EXPS}/final_best_search.ckp'")
    else:
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
        # optimizer = torch.optim.Adam(param_dicts,
        #                              lr=args.lr,
        #                              weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(param_dicts,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = tin.get_default_scheduler(optimizer)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        arch_optimizer = torch.optim.SGD(nas_parameters,
                                         lr=args.lra,
                                         momentum=args.momentum,
                                         weight_decay=args.alpha_decay)
        arch_scheduler = tin.get_default_scheduler(arch_optimizer)
        # arch_scheduler = torch.optim.lr_scheduler.StepLR(arch_optimizer, step_size=7, gamma=0.5)
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

            if epoch > EARLYSTOP_START_EPOCH:
                search_checkpoint(epoch, metrics['val_acc'])
                if earlystop(metrics['val_acc']):
                    logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                    break

                if epoch % 2 == 0:
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

    test_metrics = evaluate(True, mixprec_model, criterion, test_dl, device)
    logger.info("Search Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("Search Test Set Accuracy: {}".format(test_metrics['acc']))

    logger.info(f"alpha summary {mixprec_model.alpha_summary()}")
    # --------------------------------------------------------------------
    # FINE-TUNING LOOP
    # --------------------------------------------------------------------
    # force hard softmax sampling so that the highest coefficients are taken for the precision
    # selection.Freeze the alpha so that they remain the same ones.
    mixprec_model.update_softmax_options(False, True, False)
    mixprec_model.freeze_alpha()

    exported_model = mixprec_model

    criterion = tin.get_default_criterion()

    net_parameters_with_wd = [
        param[1] for param in exported_model.named_net_parameters() if "clip_val" not in param[0]]
    net_parameters_without_wd = [
        param[1] for param in exported_model.named_net_parameters() if "clip_val" in param[0]]

    param_dicts = [
        {'params': net_parameters_without_wd, 'weight_decay': 0},
        {'params': net_parameters_with_wd}]
    # optimizer = torch.optim.Adam(param_dicts,
    #                              lr=args.lr,
    #                              weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_dicts,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = tin.get_default_scheduler(optimizer)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

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
    logger.info("Fine-tuning Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("Fine-tuning Test Set Accuracy: {}".format(test_metrics['acc']))

    model.eval()
    logger.info("model size: {}".format(mixprec_model.get_regularization_loss()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--strength', type=float, help='Regularization Strength')
    parser.add_argument('--incremental_strength', action='store_true',
                        help='Apply increasing strength training scheme')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs',
                        default=50)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
    parser.add_argument('--temperature', type=float,
                        help='Initial temperature value', default=1.)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lra', '--learning-rate-alpha', default=1e-3, type=float,
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
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size to be used for training')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience during training')
    parser.add_argument('--w_precisions', type=str,
                        help='A string which can be parsed as tuple containing \
                        the weight precisions to be considered during the search phase')
    parser.add_argument('--a_precisions', type=str,
                        help='A string which can be parsed as tuple containing \
                        the activations precisions to be considered during the search phase')
    parser.add_argument('--save-dir-base', type=str, help='Base path to save the experiments\
                        results')
    parser.add_argument('--search-type', type=str, default='channel',
                        help='Type of the search. Currently supported: "channel", "layer"')
    parser.add_argument('--pit_checkpoint', type=str, help="Path PIT folder to load")
    args = parser.parse_args()
    main(args)
