'''

The following code is to perform self-supervised pretraining on the target dataset.

Two stages of pretraining are performed:
1. Stage 1: pretrain the SwinUNETR backbone with self-supervised learning (SSL) on the UKB (large scale) dataset.

2. Stage 2: entails further self-pretraining on the target dataset (such as the BraTS dataset), building upon the SSL groundwork laid in Stage 1
    Here we only need to use args.resume to load the pretrained model from Stage 1, and then continue training on the target dataset.


it is modified from the code main_T1T2_Stage_2.py for use on the ATLAS Dataset

1. change the inpput channel to 1
2. change dataloader to load ATLAS 2.0  dataset

For more information about the dataset see the readme on /red

For more information about the dataloader, see the file in this directory ATLASDataLoader.py

Last edited:
Dec 14, 2023
'''

import argparse
import os
from time import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_loader, get_T1T2_dataloaders
from utils.ops import aug_rand, rot_rand
import torch.nn as nn
import json

def main():

    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler, count_epoch):
        # train an epoch
        model.train()
        loss_train = []
        loss_train_recon = []
        loss_train_rot = []
        loss_train_contrastive = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            # x = batch["image"].cuda()
            x = batch["image"].to(args.device)            
            
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment

            if global_step <= 1 and args.rank == 0: #YY
                print("x:", x.size())
                print("x1 : ", x1.size(), " rot1 : ", rot1.size())
                print("x2 : ", x2.size(), " rot2 : ", rot2.size())
                print("x1_augment:", x1_augment.size())
                print("x2_augment:", x2_augment.size())

            with autocast(enabled=args.amp):
                # print(args.device)
                x1_augment = x1_augment.to(args.device)
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment) # model out1
                x2_augment = x2_augment.to(args.device)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment) # model out2
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)         # rot_p?
                rots = torch.cat([rot1, rot2], dim=0)              # rots?
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)    # imgs_con, out_combined 
                imgs = torch.cat([x1, x2], dim=0)                  # x1 -> x2, in_combined
                loss, (rot_loss, contrast_loss, recon_loss) = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)


            loss_train.append(loss.item())
            loss_train_recon.append(recon_loss.item())
            loss_train_rot.append(rot_loss.item())
            loss_train_contrastive.append(contrast_loss.item())

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            
            optimizer.zero_grad()

            if args.rank == 0:
                print(f"[{args.rank}] train: " +
                        f"epoch {count_epoch}/{args.epochs - 1}, " +    #'SSLHead' object has no attribute 'epoch'
                        f"step_within_epoch {step}/{len(train_loader) - 1}, " + 
                        f"global_step {global_step}/{args.num_steps - 1}, " +                     
                        f"loss: {loss.item():.4f}, R:{rot_loss.item():.4f}, C:{contrast_loss.item():.4f}, R:{recon_loss.item():.4f} " + #YY
                        f"time: {(time() - t1):.2f}s")            

            global_step += 1

            
            val_cond = False
            if global_step % args.eval_num == 0:
                val_cond = True

            if val_cond: #YY            
                val_loss_mean, val_loss_rot_mean, val_loss_contrastive_mean, val_loss_recon_mean, img_list = validation(args, test_loader, count_epoch, global_step)
                
            if args.rank == 0  and val_cond: #YY
                writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)
                writer.add_scalar("train/loss_rot", scalar_value=np.mean(loss_train_rot), global_step=global_step)
                writer.add_scalar("train/loss_contrastive", scalar_value=np.mean(loss_train_contrastive), global_step=global_step)
                writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)

                writer.add_scalar("validation/loss_total", scalar_value=val_loss_mean, global_step=global_step)
                writer.add_scalar("validation/loss_rot", scalar_value=val_loss_rot_mean, global_step=global_step)
                writer.add_scalar("validation/loss_contrastive", scalar_value=val_loss_contrastive_mean, global_step=global_step)
                writer.add_scalar("validation/loss_recon", scalar_value=val_loss_recon_mean, global_step=global_step)
                
                writer.add_image("validation/x1_gt", img_list[0], global_step, dataformats="HW")
                writer.add_image("validation/x1_aug", img_list[1], global_step, dataformats="HW")
                writer.add_image("validation/x1_recon", img_list[2], global_step, dataformats="HW")
                

                if val_loss_recon_mean < val_best:
                    val_best = val_loss_recon_mean
                    checkpoint = {
                        "global_step": global_step,
                        "epoch": count_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_ckp(checkpoint, os.path.join(logdir_path,  "model_bestValRMSE.pt"))

                    print(f"[{args.rank}] " + "train: Model was saved! " +
                        f"Best Recon. Val Loss {val_best:.4f} " +  
                        f"Recon. Val Loss {val_loss_recon_mean:.4f}"
                    )                     
                else:

                    print(f"[{args.rank}] " + "train: Model was not saved! " +
                        f"Best Recon. Val Loss {val_best:.4f} " +  
                        f"Recon. Val Loss {val_loss_recon_mean:.4f}"
                    )                      
        return global_step, loss, val_best



    def validation(args, test_loader, count_epoch, global_step):

        model.eval()
        loss_val = []
        loss_val_recon = []
        loss_val_rot = []
        loss_val_contrastive = []

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                # val_inputs = batch["image"].cuda()
                val_inputs = batch["image"].to(args.device)                
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(enabled=args.amp):
                    x1_augment = x1_augment.to(args.device)
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                    x2_augment = x2_augment.to(args.device)
                    rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)
                    loss, (rot_loss, contrast_loss, recon_loss) = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

                loss_val.append(loss)                # YY change 7-17-2023
                loss_val_rot.append(rot_loss)               # YY change 7-17-2023
                loss_val_contrastive.append(contrast_loss)  # YY change 7-17-2023
                loss_val_recon.append(recon_loss)           # YY change 7-17-2023

                
                x_gt = x1.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt[0][0][:, :, 48] * 255.0
                xgt = xgt.astype(np.uint8)
                x1_augment = x1_augment.detach().cpu().numpy()
                x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
                x_aug = x1_augment[0][0][:, :, 48] * 255.0
                x_aug = x_aug.astype(np.uint8)
                rec_x1 = rec_x1.detach().cpu().numpy()
                rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
                recon = rec_x1[0][0][:, :, 48] * 255.0
                recon = recon.astype(np.uint8)
                img_list = [xgt, x_aug, recon]

                if args.rank == 0:
                    print(f"[{args.rank}] " + "validation: " +
                          f"epoch {count_epoch}/{args.epochs - 1}, " +  
                          f"global_step {global_step}/{args.num_steps - 1}, " +  
                          f"Validation step {step}/{len(test_loader)}, " #YY  
                      ) 

        # YY
        if args.device != torch.device("cpu"):
            torch.cuda.synchronize(args.device)        

        # YY
        loss_val_mean = torch.sum(torch.stack(loss_val), dim=0)
        loss_val_rot_mean = torch.sum(torch.stack(loss_val_rot), dim=0) ## YY 7-17-2023
        loss_val_contrastive_mean = torch.sum(torch.stack(loss_val_contrastive), dim=0) # YY 7-17-2023
        loss_val_recon_mean = torch.sum(torch.stack(loss_val_recon), dim=0)

        # YY
        dist.all_reduce(loss_val_mean, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(loss_val_rot_mean, op=torch.distributed.ReduceOp.SUM)  ## YY 7-17-2023
        dist.all_reduce(loss_val_contrastive_mean, op=torch.distributed.ReduceOp.SUM)   ## YY 7-17-2023
        dist.all_reduce(loss_val_recon_mean, op=torch.distributed.ReduceOp.SUM)


        # YY mean 7-17-2023
        loss_val_mean = loss_val_mean/args.val_ds_len
        loss_val_rot_mean = loss_val_rot_mean/args.val_ds_len
        loss_val_contrastive_mean = loss_val_contrastive_mean/args.val_ds_len
        loss_val_recon_mean = loss_val_recon_mean/args.val_ds_len

        loss_val_mean = loss_val_mean.item()
        loss_val_rot_mean = loss_val_rot_mean.item()
        loss_val_contrastive_mean = loss_val_contrastive_mean.item()
        loss_val_recon_mean = loss_val_recon_mean.item()

        #YY 7-17-2023
        if args.rank == 0:
            print(f"Validation loss: {loss_val_mean:.4f}, R:{loss_val_rot_mean:.4f}, C:{loss_val_contrastive_mean:.4f}, R:{loss_val_recon_mean:.4f}  ") ## YY 7-17-2023

        return loss_val_mean, loss_val_rot_mean, loss_val_contrastive_mean, loss_val_recon_mean, img_list


    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint---here we use the pretrained model from Stage 1")
    parser.add_argument(
        "--pretrained_model_stage1",
        default="/red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_S_GPU064_D18_H3_07-03-2023-12:09:54-1535674/model_bestValRMSE.pt",
        type=str,
        help="pretrained checkpoint directory",
    )
    parser.add_argument("--logdir", default="/mnt/runs", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--workdir", default="/mnt", type=str, help="root of working directory")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=10, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    # parse the command-line argument --local_rank, provided by torch.distributed.launch
    parser.add_argument("--local_rank", type=int, help='provided by torch.distributed.launch')
    parser.add_argument('--distributed', action='store_true', help='start distributed training')
    
    parser.add_argument("--modality", default="T1T2", type=str,  help="modality for training T1, T2 or T1T2")


    # which dataset to use for training-- UKB or BraTS
    # TODO: this is not correct usage of argparse, make this better
    parser.add_argument("--T1T2_10k", action="store_true", help="use 10K dataset")
    parser.add_argument("--T1T2_10k_mixed", action="store_true", help="use 10K dataset mixed")
    parser.add_argument("--T1T2_40k_matched", action="store_true", help="use UKB40k")
    parser.add_argument("--T1T2_target_Brats", action="store_true", help="use target dataset BraTS")
    parser.add_argument("--T1_target_ATLAS", action='store_true', help='use ATLAS as target') 
    parser.add_argument("--split_json", default="jsons/brats21_folds.json",type=str,
                        help="the json file has the location of the images and how to split them into training testing" )
    #target data path
    parser.add_argument("--target_data_path", default="/red/ruogu.fang/share/atlas/decrypt",type=str,
                        help="target dataset folder" )
    #which fold to use for validat
    # ion
    parser.add_argument("--target_data_fold", default=0,type=int,
                        help="BRATS has 5 folds and which fold should be used for validation and the rest of them will be used for training" )

    parser.add_argument("--num_swin_block", default=2, type=int, help="number of Swin Transformer Block in layer 3")

    parser.add_argument("--num_heads_first_stage", default=3, type=int, help="number of heads in the first stage of SwinEncoder")

    parser.add_argument("--bottleneck_depth", default=768, type=int, help="depth in the last stage of SwinEncoder (bottleneck)")

    # For loading T1_T2_folds.json file
    #used for loading the UKB dataset T1 or T2 modality
    parser.add_argument("--t1_path", default="/red/ruogu.fang/share/UKB/data/Brain/20252_T1_NIFTI/T1_unzip",type=str, help="T1 Dataset folder")
    parser.add_argument("--t2_path", default="/red/ruogu.fang/share/UKB/data/Brain/20253_T2_NIFTI/T2_unzip",type=str, help="T2 Dataset folder")


    args = parser.parse_args()
    
    args.num_swin_blocks = [2,2,2,2]    
    if args.num_swin_block == 2:  args.num_swin_blocks_per_stage = [2,2,2,2]
    if args.num_swin_block == 6:  args.num_swin_blocks_per_stage = [2,2,6,2]
    if args.num_swin_block == 18: args.num_swin_blocks_per_stage = [2,2,18,2]
    
    args.num_heads_per_stage = [3, 6, 12, 24]
    if args.num_heads_first_stage == 3:  args.num_heads_per_stage = [3, 6, 12, 24]
    if args.num_heads_first_stage == 4:  args.num_heads_per_stage = [4, 8, 16, 32]
    if args.num_heads_first_stage == 6:  args.num_heads_per_stage = [6, 12, 24, 48]
    
    # assertEqual(args.feature_size * 2**4 , args.bottleneck_depth) # For 4 stage layer, feature size is related to bottleneck depth. 

    args.amp = not args.noamp    

    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    if args.distributed:
        # parameters used to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")        

        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=10))
        args.world_size = dist.get_world_size()                                
        args.rank = dist.get_rank()    
        args.device = torch.device(f"cuda:{args.local_rank}")

    else:
        # print("Training with a single process on 1 GPUs.")
        print(f"[{os.getpid()}] single-GPU training")
        args.rank = 0         
        args.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    assert args.rank >= 0

    torch.cuda.set_device(args.device)         
    print(f"[{args.rank}] current gpu: {torch.cuda.current_device()}")

    from datetime import datetime #YY
    
    if not args.distributed: args.world_size = 1

    logdir_path = os.path.join(args.logdir,
                               f"run_{os.environ['SLURM_JOB_NAME']}_GPU{args.world_size:03d}_D{args.num_swin_block}_H{args.num_heads_first_stage}_" 
                               + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") # YY 
                              )
    # print(logdir_path)

    if args.rank == 0:
        os.makedirs(logdir_path, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir_path)
        print(f"[{args.rank}] " + f"Writing Tensorboard logs to {logdir_path}")
    else:
        writer = None

    model = SSLHead(args)
    # model.cuda()
    model.to(args.device)
  
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.resume:

        print(f"[{args.rank}] " + f"Loading checkpoint from {args.pretrained_model_stage1}")
        # Access the PatchEmbed module within SwinViT
        patch_embed_layer = model.swinViT.patch_embed

        # Create a new convolutional layer with 4 input channels for 3D data
        new_proj = nn.Conv3d(2, patch_embed_layer.embed_dim, kernel_size=patch_embed_layer.patch_size,
                             stride=patch_embed_layer.patch_size)

        # Initialize the weights for the new channels
        with torch.no_grad():
            # Get the original weights
            original_weights = patch_embed_layer.proj.weight.clone()

            # Modify only the weights for the additional channels as needed
            # For example, re-initialize weights for channels 3 and 4
            nn.init.kaiming_normal_(original_weights[:, 2:4, :, :, :], mode='fan_out', nonlinearity='relu')

            # Assign the modified weights back to the layer
            patch_embed_layer.proj.weight = nn.Parameter(original_weights)

        # Replace the original proj layer with the new layer
        patch_embed_layer.proj = new_proj

        # Load the pre-trained model weights
        checkpoint = torch.load(args.pretrained_model_stage1)
        pretrained_state_dict = checkpoint['state_dict']

        # Prepare a new state dictionary for SwinUNETR's SwinViT part
        new_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith('module.swinViT.'):
                new_key = k.replace('module.swinViT.', '')  # Remove the prefix
                # Skip loading weights for the PatchEmbed proj layer
                if new_key != 'patch_embed.proj.weight' and new_key != 'patch_embed.proj.bias':
                    new_state_dict[new_key] = v

        # Load the pre-trained weights into SwinUNETR's SwinViT
        # Use strict=False to allow for the discrepancy in the first layer
        model.swinViT.load_state_dict(new_state_dict, strict=False)
        #put model to device
        model.to(args.device)


    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    loss_function = Loss(args.batch_size * args.sw_batch_size, args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # train_loader, test_loader = get_loader(args)
    train_loader, test_loader = get_T1T2_dataloaders(args)

    # torch.numel() Returns the total number of elements in the input tensor                            
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
    # print('Total parameters count', pytorch_total_params)
    if args.rank == 0:
        print(f"[{args.rank}] " + f"Total parameters count: {pytorch_total_params}")  


    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    count_epoch = 0
    while global_step < args.num_steps:     
        # train 1 epoch
        #if args.rank == 0: print(f"[{args.rank}] " + f"while loop: new epoch, global_step {global_step} ------------------------- ")
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler, count_epoch)
        count_epoch = count_epoch + 1

    checkpoint = {
        "global_step": global_step, 
        "epoch": count_epoch, 
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict()
    }

    if args.rank == 0:
        print(f"[{args.rank}] " + f"Training Finished! Best val: {best_val}") 
        save_ckp(checkpoint, os.path.join(logdir_path,  "model_final_epoch.pt"))
        print(f"[{args.rank}] " + "Saved model_final_epoch.pt")

    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
