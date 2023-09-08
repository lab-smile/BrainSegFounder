import logging

from monai.data import decollate_batch
from AverageMeter import AverageMeter
import torch
import time
from monai.data import DataLoader
from typing import Callable
import numpy as np
from utils import save_checkpoint

logger = logging.getLogger()
train_logger = logging.getLogger('training')
val_logger = logging.getLogger('validation')


def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                loss_func: Callable,
                batch_size: int,
                device: torch.device,
                max_epochs: int):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        # TODO: use a correctly set-up inferer to split this into separate labels
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        logger.debug(f'running loss: {run_loss}')
        logger.info(f'Training {epoch + 1}/{max_epochs}, {idx + 1}/{len(loader)}')
        logger.info(f'    Loss: {run_loss.avg}')
        logger.info(f'    Time: {time.time() - start_time}')
        train_logger.info(f'{epoch + 1},{idx + 1},{run_loss.avg},{time.time() - start_time}')
        start_time = time.time()
    return run_loss.avg


def validate_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        epoch: int,
        acc_func: Callable,
        max_epochs: int,
        device: torch.device,
        model_inferer=None,
        post_sigmoid=None,
        post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            logger.debug(f"Validation {epoch + 1}/{max_epochs} {idx + 1}/{len(loader)}")
            logger.debug(f"    Dice Value:")
            logger.debug(f"        Tumor  Core - {dice_tc}")
            logger.debug(f"        Enhnc Tumor - {dice_et}")
            logger.debug(f"        Whole Tumor - {dice_wt}")
            logger.debug(f"    Time: {time.time() - start_time}")
            start_time = time.time()
    return run_acc.avg


def trainer(model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_func: Callable,
            acc_func: Callable,
            scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
            max_epochs: int,
            batch_size: int,
            device: torch.device,
            output_dir: str,
            model_inferer=None,
            start_epoch=0,
            post_sigmoid=None,
            post_pred=None,
            rank=0):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    train_logger.info('epoch,img_num,loss,time')
    val_logger.info('epoch,avg,dice_tc,dice_et,dice_wt,time')
    for epoch in range(start_epoch, max_epochs):
        logger.info(time.time(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 batch_size=batch_size,
                                 device=device,
                                 max_epochs=max_epochs)
        logger.info(f'Final training {epoch + 1}/{max_epochs} loss: {train_loss} time: {time.time() - epoch_time}')

        loss_epochs.append(train_loss)
        trains_epoch.append(int(epoch))
        epoch_time = time.time()
        val_acc = validate_epoch(model,
                                 val_loader,
                                 epoch=epoch,
                                 max_epochs=max_epochs,
                                 device=device,
                                 acc_func=acc_func,
                                 model_inferer=model_inferer,
                                 post_sigmoid=post_sigmoid,
                                 post_pred=post_pred)
        dice_tc = val_acc[0]
        dice_wt = val_acc[1]
        dice_et = val_acc[2]
        val_avg_acc = np.mean(val_acc)
        logger.info(f"Final validation stats {epoch + 1}/{max_epochs}")
        logger.info(f"Dice Value:")
        logger.info(f"           Avg - {val_avg_acc}")
        logger.info(f"   Tumor  Core - {dice_tc}")
        logger.info(f"   Enhnc Tumor - {dice_et}")
        logger.info(f"   Whole Tumor - {dice_wt}")
        logger.info(f"Time: {time.time() - epoch_time}")
        val_logger.info(f'{epoch + 1},{val_avg_acc},{dice_tc},{dice_et},{dice_wt},{time.time() - epoch_time}')
        dices_tc.append(dice_tc)
        dices_wt.append(dice_wt)
        dices_et.append(dice_et)
        dices_avg.append(val_avg_acc)
        if val_avg_acc > val_acc_max and rank == 0:
            logger.info(f"New best acc ({val_acc_max} --> {val_avg_acc}). ")
            val_acc_max = val_avg_acc
            save_checkpoint(
                model,
                epoch,
                dir_add=output_dir,
                best_acc=val_acc_max,
            )
        scheduler.step()
    logger.info(f"Finetune finished! Best Accuracy: {val_acc_max}")
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )
