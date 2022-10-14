from tqdm import tqdm
from utils import tf
import torch
from configparser import SectionProxy
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Tuple
from monai.transforms import Compose


def train_epoch(model: torch.nn.Module, training_loader: DataLoader,
                optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                loss_fn: callable, settings: SectionProxy, slice_transform):
    running_loss = 0.
    for i, original_images in enumerate(tqdm(training_loader)):
        original_images, transformed_images = transform_images(original_images, settings, slice_transform)
        optim.zero_grad()
        pred = model(transformed_images)
        loss = loss_fn(pred, original_images)
        loss.backward()
        optim.step()
        running_loss += loss.item()
    scheduler.step()
    return running_loss / len(training_loader)


def validate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   loss_fn: callable,
                   settings: SectionProxy,
                   slice_transform: Compose):
    running_loss = 0
    for i, val_images in enumerate(loader):
        val_images, val_transform = transform_images(val_images, settings, slice_transform)
        pred = model(val_transform)
        loss = loss_fn(pred, val_images)
        running_loss += loss
    avg_val_loss = running_loss / len(loader)
    return avg_val_loss


def transform_images(original_images: list[torch.Tensor],
                     settings: SectionProxy, slice_transform: Compose) -> Tuple[torch.Tensor, torch.Tensor]:
    transformed_images = [deepcopy(orig_img) for orig_img in original_images]
    for j, img in enumerate(original_images):
        orig_img, tf_img = tf.transform_image(img[0, 0, :, :, :], settings, slice_transform)
        transformed_images[j][0, 0, :, :, :] = tf_img
        original_images[j][0, 0, :, :, :] = orig_img

    original_images = torch.squeeze(original_images, dim=0).cuda()
    transformed_images = torch.stack(transformed_images)
    transformed_images = torch.squeeze(transformed_images, dim=0).cuda()
    return original_images, transformed_images
