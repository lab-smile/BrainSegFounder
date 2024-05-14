import os

import bidsio
import monai.networks.nets
import numpy as np
from monai.losses import DiceLoss
from torch.cuda.amp import GradScaler, autocast

from training.lr_scheduler import WarmupCosineScheduler
from data.split_data import get_split_indices
import argparse
import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to configure and run a model with specified parameters.')

    # Paths and directories
    parser.add_argument('--checkpoint', type=str, required=True, help='Directory to pretrained model.')
    parser.add_argument('--logdir', type=str, required=True, help='Directory to save logs.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data.')
    parser.add_argument('--output', type=str, required=True, help='Directory for saving models.')

    # Basic settings
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--seed', type=int, help='Optional seed for data split')

    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training on multi-GPU.')
    parser.add_argument('--url', type=str, default='tcp://localhost:23456',
                        help='URL used to set up distributed training.')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend.')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training.')

    # Model specific parameters
    parser.add_argument('--roi', nargs=3, type=int, help='Resize the input data to these dimensions (x, y, z).')
    parser.add_argument('--in_channels', type=int, required=True, help='Number of input channels.')
    parser.add_argument('--out_channels', type=int, required=True, help='Number of output channels.')
    parser.add_argument('--feature_size', type=int, help='Size for patch embedding features.')
    parser.add_argument('--depths', nargs=4, type=int, help='Number of SSL attention heads by layer.')
    parser.add_argument('--heads', nargs=4, type=int, help='Number of heads.')

    # Training settings
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adam',
                        help='Type of optimizer to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Max gradient norm for gradient clipping.')
    parser.add_argument('--lr_decay', type=float,  help='Learning rate decay factor per epoch.')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision.')
    return parser.parse_args()


def lr(epoch: int, decay: float):
    return


def trainer(gpu: int, idxs: tuple[list[int]], dataset: bidsio.BIDSLoader, arguments: argparse.Namespace,
            distributed: bool = True, backend: str = 'nccl',
            url: str = 'tcp://localhost:23456', total_gpus: int = 1):
    if distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
        rank = gpu
        torch.distributed.init_process_group(backend=backend,
                                             init_methods=url,
                                             world_size=total_gpus,
                                             rank=rank)
    else:
        rank = gpu

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    print(f'Setup distributed training on GPU {gpu} (rank {rank})') if distributed else \
        print('Setup single-GPU training')

    if rank == 0:
        print(f'Training with batch size: {arguments.batch_size} for {arguments.epochs}')

    model = monai.networks.nets.SwinUNETR(in_channels=arguments.in_channels,
                                          out_channels=arguments.out_channels,
                                          feature_size=arguments.feature_size,
                                          use_checkpoint=True,
                                          depths=arguments.depths,
                                          num_heads=arguments.heads,
                                          drop_rate=arguments.dropout_rate)

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

    scheduler = None

    if arguments.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    scaler = GradScaler() if arguments.amp else None
    loss_function = DiceLoss()
    best_loss = 1e8

    train_idx, val_idx = idxs
    total_batches = (len(train_idx) + arguments.batch_size - 1) // arguments.batch_size

    per_gpu = total_batches // total_gpus
    leftovers = total_batches % total_gpus

    start_batch = gpu * per_gpu + min(gpu, leftovers)
    if gpu < leftovers:
        per_gpu += 1

    end_batch = start_batch + per_gpu
    batched_indices = [idxs[i * arguments.batch_size:(i + 1) * arguments.batch_size] for i in range(total_batches)]

    # Select the batches that are assigned to the current GPU
    gpu_batches = batched_indices[start_batch:end_batch]

    print(f'[GPU: {rank}] Starting training!')
    training_losses = []
    validation_losses = []
    for epoch in range(arguments.epochs):
        print(f'[GPU:{rank}] Epoch {epoch}/{arguments.epochs}...')
        model.train()
        training_loss = []
        for batch in gpu_batches:
            image, label = dataset.load_batch(batch)
            image = torch.Tensor(image, device=gpu)
            label = torch.Tensor(label, device=gpu)

            with autocast(enabled=arguments.amp):
                preds = model(image)
                loss = loss_function(preds, label)

        training_loss.append(loss)
        if arguments.amp:
            scaler.scale(loss).backward()
            optimizer.step()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
        if arguments.lr_decay:
            scheduler.step()
        optimizer.zero_grad()

        if epoch % 5 == 4 and gpu == 0:
            print('Validating on GPU 0')
            validation_loss = []
            model.eval()
            for index in val_idx:
                image, label = train_loader.load_sample(index)
                image = torch.Tensor(image, device=gpu)
                label = torch.Tensor(label, device=gpu)

                pred = model(image)
                val_loss = loss_function(label, pred)

                validation_loss.append(val_loss.item())
            validation_loss = np.mean(validation_loss)
            validation_losses.append(validation_loss)
            print(f'Validation loss: {validation_loss}')
            if validation_loss < best_loss:
                torch.save(model, os.path.join(arguments.output, 'finetune_best_val_loss.pt'))
                best_loss = validation_loss

        total_loss = torch.tensor(training_loss)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        training_losses.append(total_loss / len(train_loader))

    print(f'{training_losses=}')
    print(f'Finetuning complete! Best validation loss: {best_loss}')
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    [os.makedirs(path, exist_ok=True) for path in [args.logdir, args.output]]
    train_loader = bidsio.BIDSLoader(data_entities=[{'subject': '',
                                                     'session': '',
                                                     'suffix': 'T1w',
                                                     'space': 'MNI152NLin2009aSym'}],
                                     target_entities=[{'suffix': 'mask',
                                                       'label': 'L',
                                                       'desc': 'T1lesion'}],
                                     data_derivatives_names=['ATLAS'],
                                     target_derivatives_names=['ATLAS'],
                                     batch_size=2,
                                     root_dir='data/train/')

    indices = get_split_indices(train_loader, split_fraction=0.8, seed=args.seed)
    if args.distributed:
        n_gpus = torch.cuda.device_count()
        print(f'Found {n_gpus} accessible on each node.')
        world_size = n_gpus * args.num_nodes
        torch.multiprocessing.spawn(trainer, nprocs=n_gpus,
                                    args=(indices,
                                          train_loader, args, args.distributed,
                                          args.backend, args.url, world_size))
    else:
        trainer(0, indices, train_loader, args, args.distributed, args.backend, args.url, 1)
