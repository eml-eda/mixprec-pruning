import argparse
import datetime
import logging
import math
import os
import pathlib
# import sys
from typing import Dict, cast

import pytorch_benchmarks.keyword_spotting as kws
import torch
import torch.nn as nn
from pytorch_benchmarks.utils import (AverageMeter, CheckPoint, EarlyStopping,
                                      accuracy, seed_all)
from pytorch_model_summary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from plinio.methods import PIT

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

# torch.autograd.set_detect_anomaly(True)  # 8x slow exec

CLASS_COUNTS = torch.tensor(
    [3134., 3106., 3037., 3130., 2970., 3086., 3019., 3111., 2948., 3228., 668., 54074.]).cuda()
N_CLASSES = len(CLASS_COUNTS)
WEIGHTS_CLASSES = CLASS_COUNTS.sum() / (CLASS_COUNTS)


# Definition of the temperature annealing function
def anneal_temperature(temperature):
    # FbNetV2-like annealing
    return temperature * math.exp(-0.045)


def alpha_to_list(alpha):
    d = {}
    for key in alpha.keys():
        d[key] = alpha[key]["w_precision"].detach().cpu().tolist()
    return d


# def adjust_learning_rate(optimizer, epoch):
#     """Scheduler for the search phase"""
#     if epoch < 50:
#         lr = 1e-3
#     elif epoch < 100:
#         lr = 5e-4
#     elif epoch < 150:
#         lr = 2.5e-4
#     else:
#         lr = 1e-4
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def adjust_learning_rate_ft(optimizer, epoch):
#     """Scheduler for the fine-tuning phase"""
#     if epoch < 50:
#         lr = 1e-4
#     elif epoch < 100:
#         lr = 5e-5
#     elif epoch < 150:
#         lr = 2.5e-5
#     else:
#         lr = 1e-5
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def adjust_learning_rate(optimizer, epoch):
#     """Scheduler"""
#     if epoch < 50:
#         lr = optimizer.param_groups[0]['lr']
#     elif epoch < 100:
#         previous_lr = optimizer.param_groups[0]['lr']
#         lr = previous_lr / 2
#     elif epoch < 150:
#         previous_lr = optimizer.param_groups[0]['lr']
#         lr = previous_lr / 2
#     else:
#         previous_lr = optimizer.param_groups[0]['lr']
#         lr = previous_lr / 2.5
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch):
    """Scheduler"""
    if epoch == 50:
        previous_lr = optimizer.param_groups[0]['lr']
        lr = previous_lr / 2
    elif epoch == 100:
        previous_lr = optimizer.param_groups[0]['lr']
        lr = previous_lr / 2
    elif epoch == 150:
        previous_lr = optimizer.param_groups[0]['lr']
        lr = previous_lr / 2.5
    else:
        lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(
        search: bool,
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device,
        reg_strength: float = 0.) -> Dict[str, float]:
    """Evaluation function. If the model is in MixPrec format and thus has some architectural
    coefficients, the simple argmax operation is used to select the precision."""
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
                loss_reg = reg_strength * cast(PIT, model).get_regularization_loss()
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
                loss_reg = reg_strength * cast(PIT, model).get_regularization_loss()
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

    # (i) storage path for checkpoints & logs (ii) exp. id definition if not provided as argument
    BASE_PATH_EXPS = args.save_dir_base

    if args.exp_folder == "":
        experiment_folder = "lambda{:.2e}_epochs{}_bs{}".format(
                            args.strength,
                            args.epochs,
                            args.batch_size)
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

    logger = logging.getLogger("kws_logger")
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    logger.info(f"EXPERIMENT: <{experiment_string}>")
    logger.info("Process has PID {}".format(os.getpid()))

    logger.info(str(args))

    # check CUDA availability and ensure deterministic execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    seed_all(seed=14)

    # get the data
    datasets = kws.get_data(data_dir=args.data_dir)
    dataloaders = kws.build_dataloaders(datasets, batch_size=args.batch_size)
    train_dl, val_dl, test_dl = dataloaders

    # get the model
    model = kws.get_reference_model('ds_cnn')
    model = model.to(device)
    input_example = torch.unsqueeze(datasets[0][0][0], 0).to(device)
    input_shape = datasets[0][0][0].numpy().shape
    logger.info(summary(model, input_example, show_input=False, show_hierarchical=True))

    # --------------------------------------------------------------------
    # WARMUP LOOP
    # --------------------------------------------------------------------
    # criterion = kws.get_default_criterion()
    criterion = nn.CrossEntropyLoss(weight=WEIGHTS_CLASSES)

    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # scheduler = kws.get_default_scheduler(optimizer)
    earlystop = EarlyStopping(patience=args.patience, mode='min')

    if pathlib.Path(f'{BASE_PATH_EXPS}/final_best_warmup.ckp').exists():
        checkpoint = torch.load(f'{BASE_PATH_EXPS}/final_best_warmup.ckp')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Skipping warmup, model loaded from '{BASE_PATH_EXPS}/final_best_warmup.ckp'")
    else:
        logger.info(f"Running warmup,'{BASE_PATH_EXPS}/final_best_warmup.ckp' does not exist")
        warmup_checkpoint = CheckPoint('{}/{}/{}/warmup_checkpoints'.format(
            BASE_PATH_EXPS, experiment_folder, experiment_string), model, optimizer, 'min')

        for epoch in range(N_EPOCHS):
            metrics = train_one_epoch(
                epoch, False, model, criterion, optimizer, None, train_dl, val_dl, test_dl, device)
            adjust_learning_rate(optimizer, epoch)
            # scheduler.step()

            if epoch > EARLYSTOP_START_EPOCH:
                warmup_checkpoint(epoch, metrics['val_loss'])
                if earlystop(metrics['val_loss']):
                    logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                    break

            if epoch == (N_EPOCHS - 1):
                logger.info("Stopped at epoch {} because of maximum epochs limit".format(epoch))

        warmup_checkpoint.load_best()
        warmup_checkpoint.save(f'{BASE_PATH_EXPS}/final_best_warmup.ckp')

    # perform evaluation, either if the model has been loaded from a previous checkpoint
    # or has been trained
    test_metrics = evaluate(False, model, criterion, test_dl, device)
    logger.info("Warmup Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("Warmup Test Set Accuracy: {}".format(test_metrics['acc']))

    # --------------------------------------------------------------------
    # SEARCH LOOP
    # --------------------------------------------------------------------
    # Convert the model to the MixPrec format to perform the search
    pit_model = PIT(model, input_shape=input_shape)
    pit_model = pit_model.to(device)

    # group model/architecture parameters
    net_parameters = [param[1] for param in pit_model.named_net_parameters()]
    nas_parameters = [param[1] for param in pit_model.named_nas_parameters()]

    # criterion = kws.get_default_criterion()
    criterion = nn.CrossEntropyLoss(weight=WEIGHTS_CLASSES)
    logger.info(f"Weights for the Cross-Entropy Loss: {WEIGHTS_CLASSES}")

    # define both the net and the nas parameters' optimizers
    optimizer = torch.optim.Adam(net_parameters,
                                 lr=0.001,
                                 weight_decay=args.weight_decay)
    # scheduler = kws.get_default_scheduler(optimizer)

    arch_optimizer = torch.optim.Adam(nas_parameters,
                                      lr=0.001,
                                      weight_decay=args.alpha_decay)
    # arch_scheduler = kws.get_default_scheduler(arch_optimizer)
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
        BASE_PATH_EXPS, experiment_folder, experiment_string), pit_model, optimizer, 'min')
    earlystop = EarlyStopping(patience=args.patience, mode='min')

    for epoch in range(N_EPOCHS):
        if args.incremental_strength:
            reg_strength = min(LAMBDA / 100 + increment_lambda * epoch, LAMBDA)

        logger.info("Epoch {}, lambda = {}".format(
            epoch, reg_strength))
        metrics = train_one_epoch(
            epoch, True, pit_model, criterion, optimizer, arch_optimizer,
            train_dl, val_dl, test_dl, device, reg_strength=reg_strength)
        logger.info("architectural summary:")
        logger.info(pit_model)
        logger.info(f"model size: {pit_model.get_size()}")
        logger.info(f"model sbinarized: {pit_model.get_size_binarized()}")

        if epoch > EARLYSTOP_START_EPOCH:
            search_checkpoint(epoch, metrics['val_loss'])
            if earlystop(metrics['val_loss']):
                logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                break

            # if epoch % 25 == 0:
            #     logger.info("alpha summary", pit_model.alpha_summary(), "end alpha summary")

            if epoch == (N_EPOCHS - 1):
                logger.info("Stopped at epoch {} because of maximum epochs limit".format(epoch))

        # scheduler.step()
        # arch_scheduler.step()
        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(arch_optimizer, epoch)
        # temperature = anneal_temperature(temperature)
        # pit_model.update_softmax_temperature(temperature)

    # load and evaluate the best model found during the search phase
    logger.info("Load best model from '{}'".format(search_checkpoint.best_path))
    search_checkpoint.load_best()
    search_checkpoint.save('{}/{}/{}/final_best_search.ckp'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string))

    logger.info("final architectural summary:")
    logger.info(pit_model)
    logger.info(f"model size: {pit_model.get_size()}")
    logger.info(f"model sbinarized: {pit_model.get_size_binarized()}")

    test_metrics = evaluate(True, pit_model, criterion, test_dl, device)
    logger.info("Search Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("Search Test Set Accuracy: {}".format(test_metrics['acc']))

    # logger.info(f"alpha summary {pit_model.alpha_summary()}")
    # --------------------------------------------------------------------
    # FINE-TUNING LOOP
    # --------------------------------------------------------------------
    # Convert MixPrec model into pytorch model
    exported_model = pit_model.arch_export()
    exported_model = exported_model.to(device)
    print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))

    # criterion = kws.get_default_criterion()
    criterion = nn.CrossEntropyLoss(weight=WEIGHTS_CLASSES)

    # net_parameters = [param[1] for param in exported_model.named_parameters()]
    optimizer = kws.get_default_optimizer(exported_model)
    # optimizer = torch.optim.Adam(net_parameters, lr=5e-4, weight_decay=1e-4)
    # scheduler = kws.get_default_scheduler(optimizer)

    finetuning_checkpoint = CheckPoint('{}/{}/{}/finetuning_checkpoints'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string), exported_model, optimizer, 'min')
    earlystop = EarlyStopping(patience=args.patience, mode='min')

    for epoch in range(N_EPOCHS):
        logger.info("Epoch {}".format(epoch))
        ft_metrics = train_one_epoch(
            epoch, False, exported_model, criterion, optimizer, None,
            train_dl, val_dl, test_dl, device)

        if epoch > EARLYSTOP_START_EPOCH:
            finetuning_checkpoint(epoch, ft_metrics['val_loss'])
            if earlystop(ft_metrics['val_loss']):
                logger.info("Stopped at epoch {} because of early stopping".format(epoch))
                break

            if epoch == (N_EPOCHS - 1):
                logger.info("Stopped at epoch {} because of maximum epochs limit".format(epoch))

        adjust_learning_rate(optimizer, epoch)
        # scheduler.step()
    # logger.info(f"alpha summary {pit_model.alpha_summary()}")

    # load and evaluate the best model found during the fine-tuning phase
    logger.info("Load best model from '{}'".format(finetuning_checkpoint.best_path))
    finetuning_checkpoint.load_best()
    finetuning_checkpoint.save('{}/{}/{}/final_best_finetuning.ckp'.format(
        BASE_PATH_EXPS, experiment_folder, experiment_string))
    test_metrics = evaluate(False, exported_model, criterion, test_dl, device)
    logger.info("Fine-tuning Test Set Loss: {}".format(test_metrics['loss']))
    logger.info("Fine-tuning Test Set Accuracy: {}".format(test_metrics['acc']))
    logger.info("model sbinarized: {}".format(pit_model.get_size_binarized()))
    torch.save(
        model,
        os.path.join(BASE_PATH_EXPS, experiment_folder, experiment_string, "saved_model.ckp")
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--strength', type=float, help='Regularization Strength')
    parser.add_argument('--incremental_strength', action='store_true',
                        help='Apply increasing strength training scheme')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs',
                        default=200)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Directory with Training Data')
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
    parser.add_argument('--exp-string', default="")
    parser.add_argument('--exp-folder', default="")
    parser.add_argument('--pruning_loss', action='store_true',
                        help='Whether to add a pruning loss to the training loss function')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size to be used for training')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience during training')
    parser.add_argument('--save-dir-base', type=str, help='Base path to save the experiments\
                        results')
    args = parser.parse_args()
    main(args)
