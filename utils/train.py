from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def train_epoch(model: torch.nn.Module, loader: DataLoader,
                optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                loss_fn: callable):
    running_loss = 0.
    for i, original_images, tf_images in enumerate(tqdm(loader)):
        optim.zero_grad()
        pred = model(tf_images)
        loss = loss_fn(pred, original_images)
        loss.backward()
        optim.step()
        running_loss += loss.item()
    scheduler.step()
    return running_loss / len(loader)


def validate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   loss_fn: callable):
    running_loss = 0
    for i, val_images, val_transform in enumerate(loader):
        pred = model(val_transform)
        loss = loss_fn(pred, val_images)
        running_loss += loss
    avg_val_loss = running_loss / len(loader)
    return avg_val_loss
