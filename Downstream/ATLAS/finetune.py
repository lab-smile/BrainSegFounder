import os

import bidsio
from data.split_data import get_split_indices
import argparse
import socket
from urllib.parse import urlparse
import torch
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader, SequentialSampler, RandomSampler


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

    # Training settings
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adam',
                        help='Type of optimizer to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Max gradient norm for gradient clipping.')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay factor per epoch.')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision.')
    return parser.parse_args()


def find_valid_port(url: str) -> str:
    parsed_url = urlparse(url)
    hostname, port = parsed_url.hostname, parsed_url.port

    if port is None:
        port = 0

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((hostname, port))
                sock.close()
                break
            except socket.error as e:
                print(f'Port {port} cannot be bound with error {e}. Trying next port...')
                port = 0
    return f'{parsed_url.scheme}://{parsed_url.hostname}:{port}{parsed_url.path}'


def trainer(distributed: bool = True, gpu: int = 0, backend: str = 'nccl',
            init_methods: str = 'tcp://localhost:23456'):
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

    train_indices, validation_indices = get_split_indices(train_loader, split_fraction=0.8, seed=args.seed)
    args.url = find_valid_port(args.url)





