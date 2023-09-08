import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Finetune GatorBrain model on BraTS'
    )
    parser.add_argument('-o', '--output',
                        default='./logs',
                        help='output path for logging and model.pt')

    parser.add_argument('-d', '--data_dir',
                        default='/red/ruogu.fang/brats',
                        help='location of brats data')

    parser.add_argument('-c', '--checkpoint_dir',
                        required=True,
                        help='location of GatorBrain model to finetune from')

    parser.add_argument('-r', '--roi',
                        default=(128, 128, 128),
                        type=tuple,
                        help='region of interest for MRI scans')

    parser.add_argument('-b', '--batch_size',
                        default=2,
                        help='batch size PER GPU for training')

    parser.add_argument('-f', '--fold',
                        default=0,
                        type=int,
                        help='0-indexed fold')

    parser.add_argument('-e', '--epochs',
                        default=100,
                        type=int,
                        help='Number of finetune epochs')

    parser.add_argument('-j', '--json_path',
                        default='./jsons/brats21_folds.json',
                        help='Path to JSON file with training folds')

    parser.add_argument('-w', '--num_workers',
                        default=1,
                        type=int,
                        help='Number of workers for each dataloader')

    parser.add_argument("--local_rank",
                        type=int,
                        help="node rank for distributed training")

    parser.add_argument('--single_gpu',
                        action='store_true',
                        help='flag for single-gpu training (will ignore multi-gpu settings from torchrun)')

    parser.add_argument('-v', '--verbose',
                        action='store_true')

    return parser.parse_args()
