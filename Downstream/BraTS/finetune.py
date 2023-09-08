import logging
from functools import partial

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR

from finetune_args import parse_args
from trainer import trainer
from utils import get_loader

logger = logging.getLogger()

# Each pretrained model needs a name here
model_hyperparameters = {
    'example.pt': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 2,
        'modalities': {'t1', 't2', 't1ce', 'flair'}  # Encoding same as json files
    },

    'batchwise10k_t1t2.pt': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 2,
        'modalities': {'t1', 'flair'}
    },

    'frozen_encoder.pt': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 2,
        'modalities': {'t1', 'flair'}
    },

    'GatorBrain_Brats_channelwise.pt': {
        'out_channels': 2,
        'batch_wise': False,
        'num_modalities': 2,
        'modalities': {'t1', 'flair'}
    },

    'GatorBrain_T1_ONLY.pt': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 1,
        'modalities': {'t1'}
    },

    'swin_t1_only.pt': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 1,
        'modalities': {'t1'}
    },

    'swin_t1t2_only.pt': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 2,
        'modalities': {'t1', 'flair'}
    },

    'SwinCT_PretrainedBrats.pt': {
        'out_channels': 4,
        'batch_wise': False,
        'num_modalities': 4,
        'modalities': {'t1', 't2', 't1ce', 'flair'}
    },

    'SwinUnetrWeights.pt': {
        'out_channels': 4,
        'batch_wise': False,
        'num_modalities': 4,
        'modalities': {'t1', 't2', 't1ce', 'flair'}
    },

}


def main_worker(args, single_gpu: True):
    if single_gpu:
        args.local_rank = 0
        rank = 0
        world_size = 1
    else:
        dist.init_process_group(backend="nccl", init_method="env://")
        if args.local_rank != 0:
            f = open(os.devnull, "w")
            sys.stdout = sys.stderr = f
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    data_dir = args.data_dir
    model_path = args.checkpoint_dir
    roi = args.roi
    batch_size = args.batch_size
    fold = args.fold
    num_epochs = args.epochs
    json_path = args.json_path
    num_workers = args.num_workers
    logger.debug("Loaded args")
    hyperparameters = model_hyperparameters[model_path.split('/')[-1]]
    sw_batch_size = hyperparameters['out_channels'] * (
        hyperparameters['num_modalities'] if hyperparameters['batch_wise'] else 1)
    logger.debug(f"Loaded hparams {hyperparameters}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA enabled GPU is necessary for this code to run, sorry")
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    logger.debug(f"Device {device} located")
    train_loader, val_loader = get_loader(batch_size, data_dir, json_path, fold, roi, num_workers_=num_workers,
                                          rank=args.local_rank, world_size=world_size)

    model = SwinUNETR(
        img_size=roi,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    ).to(device)
    logger.debug("Model sent to device")
    if not single_gpu:
        model = DistributedDataParallel(model, device_ids=[device])
        logger.debug("Model passed through DDP")
    torch.backends.cudnn.benchmark = True  # Optimizes our runtime a lot

    loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inference = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=0.5,  # To be honest, I am unsure what this parameter does - need to investigate more
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoint = torch.load(model_path)
    # Cover each of the three different "types" of saved models
    if 'optimizer_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dit'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    logger.debug("Checkpoint loaded, starting training loop")
    val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch = \
        trainer(model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_func=loss,
                acc_func=dice_acc,
                scheduler=scheduler,
                max_epochs=num_epochs,
                batch_size=batch_size,
                device=device,
                output_dir=output_dir,
                model_inferer=model_inference,
                start_epoch=0,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
                rank=rank)
    logger.info(f"Final model saved at {output_dir}/model.pt")
    if not single_gpu:
        dist.destroy_process_group()


def setup_loggers(args):
    logger.addHandler(logging.StreamHandler())
    if args.verbose:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel('INFO')


if __name__ == '__main__':
    """Example multi-gpu command line usage: 
    
    torchrun --nproc_per_node=NUM_GPUS_PER_NODE --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE 
    --master_addr="localhost" --master_port=1234 finetune.py -d /red/ruogu.fang/brats -c 
    ./models/finetune_cox_j.pt -e 200
    
    Example single-gpu command line usage: 
    python finetune.py --single_gpu -d /red/ruogu.fang/brats -c ./models/finetune_cox_j.pt -e 200
    """
    cl_args = parse_args()
    setup_loggers(cl_args)
    if not cl_args.single_gpu:
        cl_args.local_rank = os.environ['LOCAL_RANK']
        logger.debug("Spawning workers")
        main_worker(cl_args, single_gpu=False)
    else:
        main_worker(cl_args, single_gpu=True)
