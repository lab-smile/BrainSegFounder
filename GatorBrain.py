import argparse
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from GatorBrainDataset import GatorBrainDataset
from utils.tf import init_transform, TransformImage
from utils.train import Trainer
from torchinfo import summary
from torch.utils.data import DataLoader
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import numpy
import nibabel as nib
from monai.networks.nets.swin_unetr import SwinUNETR

# from utils.train import train_epoch, validate_epoch, Trainer


# Setup
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12335"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_data(args: argparse.Namespace):
    train, valid, sizes = get_subjects(args)
    slice_transform = init_transform(args)
    transform_image = TransformImage(args, slice_transform)
    f_type = args.img_type + '.nii.gz'

    train_data = GatorBrainDataset(args.data_dir, f_type, train, sizes, transform=transform_image)
    valid_data = GatorBrainDataset(args.data_dir, f_type, valid, sizes, transform=transform_image)
    print(f'Training on {len(train_data)} images and validating with {len(valid_data)} images.')
    return train_data, valid_data, sizes


def init_model(sizes: list, args: argparse.Namespace):
    per_gpu_batch_size = args.batch_size
    model = SwinUNETR(img_size=sizes,
                      in_channels=1,
                      out_channels=1)
    print('Loaded model: ')
    in_size = [per_gpu_batch_size, 1] + sizes
    print(summary(model, input_size=in_size))
    return model


def load_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    optimizer = args.optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=0.0,
                                    nesterov=False)
    else:
        raise ValueError(f'Selected optimizer must be one of ["adam", "sgd"]: got {optimizer}')
    return optimizer


def load_train_objs(args: argparse.Namespace):
    train_data, valid_data, sizes = load_data(args)
    model = init_model(sizes, args)
    optimizer = load_optimizer(model, args)
    loss_fn = torch.nn.MSELoss()
    return train_data, valid_data, model, optimizer, loss_fn


def get_subjects(args: argparse.Namespace):
    data_dir = args.data_dir
    f_type = args.img_type + '.nii.gz'
    train_fraction = args.train_fraction
    subjects = list(filter(lambda s_path: (data_dir / s_path).is_file(),
                           [subject / f_type for subject in data_dir.iterdir()]))

    train, valid = train_test_split(subjects, train_size=train_fraction)
    if args.small_train and len(train) > 1000:
        train = train[0:1000]
        valid = valid[0:round(1000 * (1 - train_fraction))]

    sample_img = list(torch.as_tensor(numpy.array(nib.load(data_dir / subjects[0]).dataobj)).size())
    sizes = [32 * (i // 32 + 1) for i in sample_img]
    return train, valid, sizes


def prepare_dataloader(dataset: GatorBrainDataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, args: argparse.Namespace):
    ddp_setup(rank, world_size)
    train_data, val_data, model, optimizer, loss_fn = load_train_objs(args)
    train_data, val_data = prepare_dataloader(train_data, args.batch_size), prepare_dataloader(val_data,
                                                                                               args.batch_size)
    trainer = Trainer(model, train_data, val_data, optimizer, loss_fn, rank)
    writer = SummaryWriter(args.log_path + '/GatorBrain')
    trainer.train(args.total_epochs, writer)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining script for GatorBrain project using UKB data')
    parser.add_argument('total_epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--batch_size', default=1, help='Batch size per device')
    parser.add_argument('--model_name', default="GB-SWINUnetR-T1_brain_to_MNI", help="Name to save model checkpoints")
    parser.add_argument('--img_type', default="T1_brain_to_MNI", help="Image (preprocessing) type for UKB images")
    parser.add_argument('--small_train', default=True, help='If true only pretrain on 1000 images')
    parser.add_argument('--data_dir', default='/red/ruogu.fang/UKB/Brain/20252_T1_NIFTI/T1_unzip',
                        help="UKB Data Directory")
    parser.add_argument('--optimizer', default='adam', help='Optimizer for training')
    parser.add_argument('--learning_rate', default=0.01, help="Training learning rate")
    parser.add_argument('--gamma', default=0.5, help='Scheduler gamma // not implemented')
    parser.add_argument('--train_fraction', default=0.8,
                        help='fraction of total UKB data to train on (rest becomes validation)')
    parser.add_argument('--flip_rate', default=0.4, help='Rate at which images are flipped along a random axis')
    parser.add_argument('--shuffling_rate', default=0.5, help='Chance a sub-volume of an image is shuffled')
    parser.add_argument('--painting_rate', default=0.9, help='Change a sub-volume of an image is painted')
    parser.add_argument('--inpainting_rate', default=0.5, help='Given a sub-volume is painted, chance it is inpainted')
    parser.add_argument('--non_linear_transformation_rate', default=0.9,
                        help='Chance a sub-volume of an image has a non-linear transformation applied')
    parser.add_argument('--num_slices', default=10, help='Number of cubic sub-volumes to generate per image')
    parser.add_argument('--window_size', default=32, help='Size (in pixels) of one side of each cubic sub-volume')
    parser.add_argument('--model_path', default='/blue/ruogu.fang/cox.j/GatorBrain/pretrained_weights',
                        help='Path to save model checkpoints and final weights to')
    parser.add_argument('--log_path', default='/blue/ruogu.fang/cox.j/GatorBrain/logs',
                        help='Path to save logs and SummaryWriter objects')

    args_ = parser.parse_args()
    world_size_ = torch.cuda.device_count()
    mp.spawn(main, args=(world_size_, args_), nprocs=world_size_)

# cfg = ConfigParser(interpolation=BasicInterpolation())
# cfg.read('config.ini')
# general_cfg = cfg['general']
# data_cfg = cfg['data']
# training_cfg = cfg['pretraining']
# tf_cfg = cfg['transformations']
# log_cfg = cfg['logging']
# for key, value in log_cfg.items():
#     if key.endswith('path') and not os.path.exists(value):
#         os.mkdir(value)
# model_name = general_cfg['model_name']
# writer = SummaryWriter(log_cfg['log_path'])


# Train/Valid split
# data_dir = pathlib.Path(data_cfg['dir'])
# f_type = general_cfg['img_type'] + '.nii.gz'
# train_fraction = training_cfg.getfloat('train_fraction')
# subjects = list(filter(lambda s_path: (data_dir / s_path).is_file(),
#                        [subject / f_type for subject in data_dir.iterdir()]))

# train, valid = train_test_split(subjects, train_size=train_fraction)
#
# if general_cfg.getboolean('small_train') and len(train) > 1000:
#     train = train[0:1000]
#     valid = valid[0:round(1000 * (1 - train_fraction))]
#
# sample_img = list(torch.as_tensor(numpy.array(nib.load(data_dir / subjects[0]).dataobj)).size())
# sizes = [32 * (i // 32 + 1) for i in sample_img]
#
# # Dataset Init
# slice_transform = init_transform(tf_cfg)
# transform_image = TransformImage(tf_cfg, slice_transform)
# train_set = GatorBrainDataset(data_dir, f_type, train, sizes, transform=transform_image)
# valid_set = GatorBrainDataset(data_dir, f_type, valid, sizes, transform=transform_image)
# print(f'Training on {len(train_set)} images and validating with {len(valid_set)} images.')
#
# # Format Model
# per_gpu_batch_size = training_cfg.getint('batch_size')
# device = torch.device('cuda')
# model = SwinUNETR(img_size=sizes,
#                   in_channels=1,
#                   out_channels=1)
# model.cuda()
#
# print(f'Using {torch.cuda.device_count()} CUDA device(s)')
# print('Loaded model: ')
# in_size = [per_gpu_batch_size, 1] + sizes
# print(summary(model, input_size=in_size))
# batch_size = per_gpu_batch_size * torch.cuda.device_count()
#
# # Pretrain Model
# print('Preparing model for training...')
# initial_epoch, max_epochs = 0, training_cfg.getint('nb_epochs')
# optimizer, lr = training_cfg['optimizer'], training_cfg.getfloat('learning_rate')
# patience = training_cfg.getint('patience')
# criterion = torch.nn.MSELoss()
#
# if optimizer == 'adam':
#     optimizer = torch.optim.Adam(model.parameters(), lr)
# elif optimizer == 'sgd':
#     optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0, nesterov=False)
# else:
#     raise ValueError(f'Selected optimizer must be one of ["adam", "sgd"]: got {optimizer}')
#
# workers = training_cfg.getint('workers')
#
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(patience * 0.8), gamma=0.5)
# training_loader = DataLoader(train_set, batch_size=batch_size,
#                              shuffle=False, num_workers)
# validation_loader = DataLoader(valid_set, batch_size=batch_size,
#                                shuffle=False, num_workers=0)
#
# train_losses = []
# valid_losses = []
# avg_train_losses = []
# avg_valid_losses = []
#
# best_loss = 100000
# model_path = None
# num_epochs_since_improve = 0
# patience = training_cfg.getint('patience')
# print("Beginning pretraining...")
#
# for epoch in range(initial_epoch, max_epochs):
#     print(f'Training epoch {epoch + 1}')
#     model.train(True)
#     avg_train_loss = train_epoch(model, training_loader, optimizer, scheduler, criterion)
#     writer.add_scalar('train_loss', avg_train_loss, epoch)
#     model.train(False)
#     with torch.no_grad():
#         print(f"Validating epoch {epoch + 1}")
#         avg_val_loss = validate_epoch(model, validation_loader, criterion)
#         print(f'LOSS validation - {avg_val_loss}, train - {avg_train_loss}')
#         writer.add_scalar('val_loss', avg_val_loss, epoch)
#     if avg_val_loss < best_loss:
#         num_epochs_since_improve = 0
#         print("New best loss.")
#         best_loss = avg_val_loss
#         model_path = f"{log_cfg['model_path']}/{model_name}_{epoch}.ckpt"
#         torch.save(model.state_dict(), model_path)
#     else:
#         num_epochs_since_improve += 1
#     if num_epochs_since_improve == patience:
#         print("I'm at my wit's end! (Patience reached - stopping)")
#         break
#
# print(f"Pretraining complete. Best model saved in {model_path}")
