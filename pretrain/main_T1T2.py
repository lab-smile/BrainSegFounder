# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment) # model out1
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

            # if args.distributed:
            #     if args.rank == 0:
            #         print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))
            # else:
            #     print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))
            # if args.rank == 0:
            if args.rank == 0:
                print(f"[{args.rank}] train: " +
                        f"epoch {count_epoch}/{args.epochs - 1}, " +    #'SSLHead' object has no attribute 'epoch'
                        f"step_within_epoch {step}/{len(train_loader) - 1}, " + 
                        f"global_step {global_step}/{args.num_steps - 1}, " +                     
                        f"loss: {loss.item():.4f}, R:{rot_loss.item():.4f}, C:{contrast_loss.item():.4f}, R:{recon_loss.item():.4f} " + #YY
                        f"time: {(time() - t1):.2f}s")            

            global_step += 1
    
            # Validate on single GPU
            # if args.distributed:
            #     val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            # else:
            #     val_cond = global_step % args.eval_num == 0
            # val_cond = (args.rank == 0) and (global_step % args.eval_num == 0)
            # val_cond = (args.rank == 0) and (global_step % args.eval_num == 0)
            
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
                    save_ckp(checkpoint, os.path.join(args.logdir,  "model_bestValRMSE.pt"))
                    # print(
                    #     "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                    #         val_best, val_loss_recon
                    #     )
                    # )
                    print(f"[{args.rank}] " + "train: Model was saved! " +
                        f"Best Recon. Val Loss {val_best:.4f} " +  
                        f"Recon. Val Loss {val_loss_recon_mean:.4f}"
                    )                     
                else:
                    # print(
                    #     "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                    #         val_best, val_loss_recon
                    #     )
                    # )
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
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
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
                # print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(step, loss, loss_recon))
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
    parser.add_argument("--resume", default=None, type=str, help="resume training")    
    parser.add_argument("--logdir", default="/mnt/runs", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--workdir", default="/mnt", type=str, help="root of working directory")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=10, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
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
    parser.add_argument("--local_rank", "--local-rank", type=int, help='provided by torch.distributed.launch')
    parser.add_argument('--distributed', action='store_true', help='start distributed training')
    
    parser.add_argument("--modality", default="T1", type=str,  help="modality for training T1, T2 or T1T2")

    parser.add_argument("--T1T2_10k", action="store_true", help="use 10K dataset")
    parser.add_argument("--T1T2_10k_mixed", action="store_true", help="use 10K dataset mixed")

    parser.add_argument("--split_json", default="jsons/GBR_T1T2_matched_image.json",type=str, help="Dataset split JSON")

    parser.add_argument("--num_swin_block", default=2, type=int, help="number of Swin Transformer Block in layer 3")

    parser.add_argument("--num_heads_first_stage", default=3, type=int, help="number of heads in the first stage of SwinEncoder")

    parser.add_argument("--bottleneck_depth", default=768, type=int, help="depth in the last stage of SwinEncoder (bottleneck)")

    # For loading T1_T2_folds.json file, the original format by Joseph
    parser.add_argument("--t1_path", default="/red/ruogu.fang/UKB/data/Brain/20252_T1_NIFTI/T1_unzip",type=str, help="T1 Dataset folder")
    parser.add_argument("--t2_path", default="/red/ruogu.fang/UKB/data/Brain/20253_T2_NIFTI/T2_unzip",type=str, help="T2 Dataset folder")

    parser.add_argument("--jobID", default="", type=str, help="run jobID to match log file with folder")    

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
    
    # args.distributed = False
    # if "WORLD_SIZE" in os.environ:
    #     args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    # args.device = "cuda:0"
    # args.world_size = 1
    # args.rank = 0

    # for debugging purpose
    if args.distributed:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 

    if args.distributed:   
        # args.device = "cuda:%d" % args.local_rank
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        # args.world_size = torch.distributed.get_world_size()
        # args.rank = torch.distributed.get_rank()

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
        # print(
        #     "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
        #     % (args.rank, args.world_size)
        # )
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

    args.logdir = os.path.join(args.logdir, 
                               f"run_{os.environ['SLURM_JOB_NAME']}_GPU{args.world_size:03d}_D{args.num_swin_block}_H{args.num_heads_first_stage}_"
                               + args.jobID
                               + "_"
                               + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") # YY 
                              )

    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[{args.rank}] " + f"Writing Tensorboard logs to {args.logdir}")
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

    # see args.checkpoint in BRATS script?, no need for new_state_dict?
    # can do this after model.to(args.device)?
    # args.epochs < resumed epoch?
    # if args.resume:  
    #     model_pth = args.resume
    #     model_dict = torch.load(model_pth)
    #     model.load_state_dict(model_dict["state_dict"])
    #     model.epoch = model_dict["epoch"]      
    #     model.optimizer = model_dict["optimizer"]
    if args.resume is not None:
        try:
            # model_dict = torch.load("./pretrained_models/model_swinvit.pt")
            model_dict = torch.load(args.resume)
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                if args.rank == 0:
                    print(f"[{args.rank}] " + "Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False to ignore non-matching key, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            if 'epoch' in model_dict:
                model.epoch = model_dict["epoch"]
            if 'optimizer' in model_dict:      
                model.optimizer = model_dict["optimizer"]            
            # print("Using pretrained self-supervised Swin UNETR backbone weights !")
            if args.rank == 0:
                print(f"[{args.rank}] " + "Using pretrained self-supervised Swin UNETR backbone weights !")            
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

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
    # model.epoch after training might not equal to args.epoch
    # checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    checkpoint = {
        "global_step": global_step, 
        "epoch": count_epoch, 
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict()
    }

    # if args.distributed:
    #     if args.rank == 0:
    #         torch.save(model.state_dict(), args.logdir + "final_model.pth")
    #     dist.destroy_process_group()
    # else:
    #     torch.save(model.state_dict(), args.logdir + "final_model.pth")
    # save_ckp(checkpoint, args.logdir + "/model_final_epoch.pt") 
    if args.rank == 0:
        print(f"[{args.rank}] " + f"Training Finished! Best val: {best_val}") 
        save_ckp(checkpoint, os.path.join(args.logdir,  "model_final_epoch.pt"))
        print(f"[{args.rank}] " + "Saved model_final_epoch.pt")

    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
