"""File for Stage 2 Pretraining of the BrainFounder model on the ATLAS v2.0 dataset
"""
import argparse
import logging
import os
from typing import Tuple, Optional

import numpy as np
import monai.transforms

from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.cuda
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset.ATLASDataset import ATLASDataset, data_entities, target_entities
from dataset.ATLASSampler import ATLASSampler
from training.lr_scheduler import WarmupCosineScheduler
from training.loss import Loss
from training.ops import rot_rand, aug_rand
from models.ssl_head import SSLHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Required arguments for this script
    parser.add_argument('--checkpoint', '-c')
    parser.add_argument('--data_dir', '-d')
    parser.add_argument('--pretrained_model', '-p', help='Path to the Stage 1 Pretrained Model')

    # Optional arguments
    parser.add_argument('--logdir', '-l', type=str, help='Directory for tensorboard and output logs.',
                        default='./logs/')
    parser.add_argument('--verbose', '-v', action='store_true', help='log debug output')
    parser.add_argument('--num_workers', '-w', default=2, type=int,
                        help='Number of workers for the dataloader. For best results, set to number of CPUs available')
    parser.add_argument('--output', '-o', type=str, help='Directory for saving output predictions and models',
                        default='./output/')
    parser.add_argument('--visual_output', action='store_true', help='If set, save random output predictions from last'
                                                                     'training set.')

    # Distributed arguments
    parser.add_argument('--distributed', action='store_true', help='If set, train on all available GPUs.')
    parser.add_argument('--url', default='tcp://127.0.0.1:23456', type=str, help='URL for distributed training.')
    parser.add_argument('--backend', default='nccl', type=str, choices=['nccl', 'gloo', 'mpi'],
                        help='PyTorch distributed backend')
    parser.add_argument('--num_nodes', '-n', default=1, type=int, help='Number of GPU nodes to train on')

    # Hyperparameters
    parser.add_argument('--batch_size', '-b', default=2, type=int, help='Batch size for each GPU')
    parser.add_argument('--epochs', '-e', default=1000, type=int, help='Number of epochs to pretrain on')
    parser.add_argument('--roi', '-r', nargs=3, type=int, default=[96, 96, 96], help='ROI (x y z) for the model')
    parser.add_argument('--lr_decay', action='store_true', help="If set, use learning rate decay")
    parser.add_argument('--max_grad_norm', default=None, type=float,
                        help='If set, normalized gradients will be clipped to this value')
    parser.add_argument('--dropout_rate', default=0.05, type=float, help='Dropout rate for SSL Head')
    parser.add_argument('--path_drop_rate', default=0.01, type=float, help='Dropout rate for entire layers in SSL head')
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer for the model')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate for optimizer')
    parser.add_argument('--num_warmup_steps', default=50, type=int,
                        help='Number of warmup epochs before learning rate reaches set value')
    parser.add_argument('--lr_scheduler', type=str, default='warmup_cosine', choices=['warmup_cosine', 'polynomial'],
                        help='Learning rate scheduler. Only used if decay is set.')
    parser.add_argument('--amp', action='store_true', help='Use PyTorch AMP for training')

    # Data and Transforms
    parser.add_argument('--in_channels', '-i', default=1, help='Number of input channels in the data')
    parser.add_argument('--feature_size', default=756, type=int, help='Feature size of patch embedding')
    parser.add_argument('--depths', nargs=4, type=int, help='Depths (by layer) for the SSL Head.',
                        default=[2, 2, 2, 2])
    parser.add_argument('--heads', nargs=4, type=int, default=[3, 6, 12, 24], help='SSL attention heads, by layer')
    return parser.parse_args()


def setup_logger(verbose: bool, logdir: str) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '[%(asctime)s - %(levelname)s] %(message)s'
    logging.basicConfig(filename=os.path.join('', logdir, 'log.txt'), level=log_level,
                        format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    
def setup_directories(paths: list[str]) -> None:
    [os.makedirs(path, exist_ok=True) for path in paths]


def trainer(gpu: int, arguments: argparse.Namespace, total_gpus: int, best_loss: int = 1e8) -> None:
    if arguments.distributed:
        mp.set_start_method('fork', force=True)
        rank = gpu
        dist.init_process_group(
            backend=arguments.backend, init_method=arguments.url, world_size=total_gpus, rank=rank
        )
    else:
        rank = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    dataset = ATLASDataset(data_entities, target_entities,
                           data_derivatives_names=['ATLAS'],
                           target_derivatives_names=['ATLAS'],
                           root_dir='data/train',
                           transform=monai.transforms.Compose([
                               monai.transforms.ToTensor(),
                               monai.transforms.Resize(spatial_size=arguments.roi)
                           ]),
                           target_transform=None)

    sampler = ATLASSampler(dataset=dataset)

    loader = DataLoader(dataset, batch_size=arguments.batch_size, num_workers=arguments.num_workers,
                        sampler=sampler, shuffle=(sampler is None), pin_memory=True)

    print(f'Setup dataloader on GPU {gpu} (rank {rank})')
    if rank == 0:
        print(f'Training with batch size: {arguments.batch_size} for {arguments.epochs} epochs.')

    model = SSLHead(spatial_dimensions=3,
                    in_channels=arguments.in_channels,
                    feature_size=arguments.feature_size,
                    dropout_rate=arguments.dropout_rate,
                    stochastic_depth_rate=arguments.path_drop_rate,
                    depths=arguments.depths,
                    heads=arguments.heads,
                    use_checkpoint=True)

    loss_function = Loss(batch_size=arguments.batch_size, device=gpu)

    if arguments.pretrained_model is not None:
        original_weights = torch.load(arguments.pretrained_model)
        state_dict = original_weights['state_dict']

        if "module." in list(state_dict.keys())[0]:
            if rank == 0:
                print(f"[{rank}] Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)

        # Not strict to only load encoder weights
        model.load_state_dict(state_dict, strict=False)
        if 'epoch' in original_weights:
            model.epoch = original_weights["epoch"]
        if 'optimizer' in original_weights:
            model.optimizer = original_weights["optimizer"]
        if rank == 0:
            print(f"[GPU:{rank}]: Using pretrained backbone weights!")

    model.to(device=gpu)
    if arguments.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=arguments.learning_rate,
                                     weight_decay=arguments.lr_decay)
    elif arguments.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=arguments.learning_rate,
                                      weight_decay=arguments.lr_decay)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=arguments.learning_rate, momentum=arguments.momentum,
                                    weight_decay=arguments.lr_decay)

    if arguments.lr_decay:
        if arguments.lr_scheduler == 'warmup_cosine':
            scheduler = WarmupCosineScheduler(optimizer, warmup_steps=arguments.num_warmup_steps,
                                              t_total=arguments.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=(lambda e:
                                                                     (1 - e / arguments.epochs) ** 0.9))
    else:
        scheduler = None

    if arguments.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        print(f'[GPU:{rank}]: Total parameters count - '
              f'{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    scaler = GradScaler() if arguments.amp else None
    for epoch in range(arguments.epochs):
        print(f'[GPU:{rank}] Epoch {epoch}/{arguments.epochs}...')
        loss, individual_losses = train(arguments, model, loss_function, global_step=epoch, gpu=rank,
                                        train_loader=loader, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        print(f'[GPU:{rank}]: Completed epoch {epoch}.')
        print(f'[GPU: {rank}]: REPORT - {loss},{len(loss)}')

        n_samples = torch.tensor(len(loss), device=gpu)
        total_loss = torch.tensor(sum(loss), device=gpu)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
        global_loss_mean = total_loss / n_samples
        if rank == 0:
            print(f'[GPU: {rank}]: Average global loss - {global_loss_mean}')
            if global_loss_mean < best_loss:
                torch.save(model, os.path.join(arguments.output, 'stage_2_best_loss.pt'))
                best_loss = global_loss_mean
    dist.destroy_process_group()


def train(arguments: argparse.Namespace, model: torch.nn.Module, loss_function: Loss,
          global_step: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer, gpu: int,
          scaler: Optional[torch.cuda.amp.GradScaler] = None,
          scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None):
    model.train()
    training_loss = []
    rotation_loss = []
    contrastive_loss = []
    reconstruction_loss = []

    if global_step == 0:
        print('Starting training!')

    for step, batch in enumerate(train_loader):
        print(f'[GPU {gpu}]: {step} / {len(train_loader)}')
        image, _ = batch  # Already on GPU from transforms
        first_augment, first_rotations = augment_image(gpu, image)
        second_augment, second_rotations = augment_image(gpu, image)
        ground_truth_images = torch.cat([first_augment, second_augment], dim=0).to(gpu)
        ground_truth_rotations = torch.cat([first_rotations, second_rotations], dim=0).to(gpu)

        with autocast(enabled=arguments.amp):
            first_prediction = model(first_augment)
            second_prediction = model(second_augment)
            predicted_rotations = torch.cat([first_prediction[0], second_prediction[0]], dim=0).to(gpu)
            predicted_images = torch.cat([first_prediction[2], second_prediction[2]], dim=0).to(gpu)

            loss, loss_by_task = loss_function(predicted_rotations, ground_truth_rotations,  # Rotation loss
                                               first_prediction[1], second_prediction[1],    # Contrastive loss
                                               predicted_images, ground_truth_images)        # Reconstructive loss

            if arguments.visual_output and global_step == arguments.epochs and step == len(train_loader) and gpu == 0:
                np.save(f'{arguments.logdir}/predictions.npy', predicted_images.numpy(force=True))
                np.save(f'{arguments.logdir}/truth.npy', ground_truth_images.numpy(force=True))

        training_loss.append(loss.item())
        rotation_loss.append(loss_by_task[0].item())
        contrastive_loss.append(loss_by_task[1].item())
        reconstruction_loss.append(loss_by_task[2].item())

        if arguments.amp:
            scaler.scale(loss).backward()
            optimizer.step()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if arguments.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), arguments.max_grad_norm)
            optimizer.step()

        if arguments.lr_decay:
            scheduler.step()
        optimizer.zero_grad()
    return training_loss, (rotation_loss, contrastive_loss, reconstruction_loss)


def augment_image(gpu: int, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rotated_image, rotations_performed = rot_rand(gpu, image)
    augmented_image = aug_rand(gpu, rotated_image)
    return augmented_image, rotations_performed


if __name__ == '__main__':
    args = parse_args()
    setup_directories([args.output, args.logdir])
    setup_logger(args.verbose, args.logdir)
    logger = logging.getLogger('ATLAS')
    if args.distributed:
        n_gpus = torch.cuda.device_count()
        print(f'Found {n_gpus} gpus accessible on each node.')
        world_size = n_gpus * args.num_nodes
        mp.spawn(trainer, nprocs=n_gpus, args=(args, n_gpus, world_size))
    else:
        trainer(gpu=0, arguments=args, total_gpus=1)

    print(f'Training complete - final model saved at {args.output}/stage_2_best_loss.pt')
