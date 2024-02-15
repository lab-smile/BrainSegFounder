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
import typing
from time import time
from datetime import datetime, timedelta
from typing import Protocol, Optional, Any, Tuple, Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch import Tensor

from losses.loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_T1T2_dataloaders
from utils.ops import aug_rand, rot_rand


class Args(Protocol):
    # Argparse Provided
    resume: Optional[str]
    logdir: str
    workdir: str
    epochs: int
    num_steps: int
    eval_num: int
    warmup_steps: int
    num_workers: int
    in_channels: int
    out_channels: int
    feature_size: int
    dropout_path_rate: float
    use_checkpoint: bool
    spatial_dims: int
    a_min: int
    a_max: int
    b_min: float
    b_max: float
    space_x: float
    space_y: float
    space_z: float
    roi_x: int
    roi_y: int
    roi_z: int
    batch_size: int
    sw_batch_size: int
    lr: float
    decay: float
    momentum: float
    lrdecay: bool
    max_grad_norm: float
    loss_type: str
    opt: str
    lr_schedule: str
    grad_clip: bool
    noamp: bool
    smartcache_dataset: bool
    cache_dataset: bool
    local_rank: Optional[int]
    distributed: bool
    modality: str
    target_data_fold: int
    T1T2_10k: bool
    T1T2_10k_mixed: bool
    split_json: str
    num_swin_block: int
    num_heads_first_stage: int
    bottleneck_depth: int
    t1_path: str
    t2_path: str
    jobID: str

    # Added during runtime
    amp: bool
    device: torch.device
    rank: int
    val_ds_len: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--logdir", default="/mnt/runs", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--workdir", default="/mnt", type=str, help="root of working directory")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=10, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
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
    parser.add_argument("--local_rank", "--local-rank", type=int, help="provided by torch.distributed.launch")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--modality", default="T1T2", type=str, help="modality for training T1, T2 or T1T2")
    parser.add_argument("--target_data_fold", default=0)
    parser.add_argument('--target_data_path', default='/red/ruogu.fang/atlas/decrypt/ATLAS_2')
    parser.add_argument("--T1T2_10k", action="store_true", help="use 10K dataset")
    parser.add_argument("--T1T2_10k_mixed", action="store_true", help="use 10K dataset mixed")
    parser.add_argument(
        "--split_json", default="jsons/GBR_T1T2_matched_image.json", type=str, help="Dataset split JSON"
    )
    parser.add_argument("--num_swin_block", default=2, type=int, help="number of Swin Transformer Block in layer 3")
    parser.add_argument(
        "--num_heads_first_stage", default=3, type=int, help="number of heads in the first stage of SwinEncoder"
    )
    parser.add_argument(
        "--bottleneck_depth", default=768, type=int, help="depth in the last stage of SwinEncoder (bottleneck)"
    )
    # For loading T1_T2_folds.json file
    parser.add_argument(
        "--t1_path",
        default="/red/ruogu.fang/UKB/data/Brain/20252_T1_NIFTI/T1_unzip",
        type=str,
        help="T1 Dataset folder",
    )
    parser.add_argument(
        "--t2_path",
        default="/red/ruogu.fang/UKB/data/Brain/20253_T2_NIFTI/T2_unzip",
        type=str,
        help="T2 Dataset folder",
    )
    parser.add_argument('--T1T2_target_ATLAS', action='store_true')
    parser.add_argument("--jobID", default="", type=str, help="run jobID to match log file with folder")
    return typing.cast(Args, parser.parse_args())


class TrainerAndValidator:
    def __init__(self, args: Args, writer: Optional[SummaryWriter] = None) -> None:
        self._args = args
        self._writer = writer
        self._model = SSLHead(self._args)
        self._model.to(self._args.device)

        if self._args.opt == "adam":
            self._optimizer = optim.Adam(
                params=self._model.parameters(), lr=self._args.lr, weight_decay=self._args.decay
            )
        elif self._args.opt == "adamw":
            self._optimizer = optim.AdamW(
                params=self._model.parameters(), lr=self._args.lr, weight_decay=self._args.decay
            )
        elif self._args.opt == "sgd":
            self._optimizer = optim.SGD(
                params=self._model.parameters(),
                lr=self._args.lr,
                momentum=self._args.momentum,
                weight_decay=self._args.decay,
            )
        else:
            raise ValueError(f"Unknown value for argument --opt: {self._args.opt}")

        if self._args.resume is not None:
            try:
                model_dict = torch.load(self._args.resume)
                state_dict = model_dict["state_dict"]
                # fix potential differences in state dict keys from pre-training to fine-tuning
                if "module." in list(state_dict.keys())[0]:
                    if self._args.rank == 0:
                        print(f"[{self._args.rank}] " + "Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)
                # We now load model weights, setting param `strict` to False to ignore non-matching key, i.e.:
                # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
                # the decoder weights untouched (CNN UNet decoder).
                self._model.load_state_dict(state_dict, strict=False)
                if "epoch" in model_dict:
                    self._model.epoch = model_dict["epoch"]
                if "optimizer" in model_dict:
                    self._model.optimizer = model_dict["optimizer"]
                if self._args.rank == 0:
                    print(f"[{self._args.rank}] " + "Using pretrained self-supervised Swin UNETR backbone weights !")
            except ValueError:
                raise ValueError("Self-supervised pre-trained weights not available for" + str(self._args.resume))

        if self._args.lrdecay:
            if self._args.lr_schedule == "warmup_cosine":
                self._scheduler = WarmupCosineSchedule(
                    self._optimizer, warmup_steps=self._args.warmup_steps, t_total=self._args.num_steps
                )
            elif self._args.lr_schedule == "poly":
                self._scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self._optimizer, lr_lambda=lambda epoch: (1 - float(epoch) / float(self._args.epochs)) ** 0.9
                )
        else:
            self._scheduler = None

        self._loss_function = Loss(self._args.batch_size * self._args.sw_batch_size, self._args)
        if self._args.distributed:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            self._model = DistributedDataParallel(self._model, device_ids=[self._args.local_rank])
        self._train_loader, self._test_loader = get_T1T2_dataloaders(self._args)

        # torch.numel() Returns the total number of elements in the input tensor
        pytorch_total_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        if self._args.rank == 0:
            print(f"[{self._args.rank}] " + f"Total parameters count: {pytorch_total_params}")

        self._global_step = 0
        self._scaler = GradScaler() if self._args.amp else None

    def train_all_steps(self) -> Tuple[Dict[str, Any], float]:
        count_epoch = 0
        best_val = 1e8
        while self._global_step < self._args.num_steps:
            loss, best_val = self._train(best_val, count_epoch)
            count_epoch += 1

        checkpoint = {
            "global_step": self._global_step,
            "epoch": count_epoch,
            "state_dict": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

        return checkpoint, best_val

    def _train(self, val_best: float, count_epoch: int):
        # train an epoch
        self._model.train()
        loss_train = []
        loss_train_recon = []
        loss_train_rot = []
        loss_train_contrastive = []

        for step, batch in enumerate(self._train_loader):
            t1 = time()
            x = batch["image"].to(self._args.device)

            x1, rot1 = rot_rand(self._args, x)
            x2, rot2 = rot_rand(self._args, x)
            x1_augment = aug_rand(self._args, x1)
            x2_augment = aug_rand(self._args, x2)

            with autocast(enabled=self._args.amp):
                rot1_p, contrastive1_p, rec_x1 = self._model(x1_augment)  # model out1
                rot2_p, contrastive2_p, rec_x2 = self._model(x2_augment)  # model out2
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)  # rot_p?
                rots = torch.cat([rot1, rot2], dim=0)  # rots?
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)  # imgs_con, out_combined
                imgs = torch.cat([x1, x2], dim=0)  # x1 -> x2, in_combined
                loss, (rot_loss, contrast_loss, recon_loss) = self._loss_function(
                    rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs
                )

            loss_train.append(loss.item())
            loss_train_recon.append(recon_loss.item())
            loss_train_rot.append(rot_loss.item())
            loss_train_contrastive.append(contrast_loss.item())

            if self._args.amp:
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                loss.backward()
                if self._args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._args.max_grad_norm)
                self._optimizer.step()

            if self._args.lrdecay:
                self._scheduler.step()

            self._optimizer.zero_grad()

            if self._args.rank == 0:
                print(
                    f"[{self._args.rank}] train: "
                    + f"epoch {count_epoch}/{self._args.epochs - 1}, "
                    + f"step_within_epoch {step}/{len(self._train_loader) - 1}, "  # 'SSLHead' object has no attribute 'epoch'
                    + f"global_step {self._global_step}/{self._args.num_steps - 1}, "
                    + f"loss: {loss.item():.4f}, R:{rot_loss.item():.4f}, C:{contrast_loss.item():.4f}, R:{recon_loss.item():.4f} "
                    + f"time: {(time() - t1):.2f}s"  # YY
                )

            self._global_step += 1
            if self._global_step % self._args.eval_num == 0:
                (
                    val_loss_mean,
                    val_loss_rot_mean,
                    val_loss_contrastive_mean,
                    val_loss_recon_mean,
                    img_list,
                ) = self._validation(count_epoch)
                if self._writer is not None:
                    self._writer.add_scalar(
                        "train/loss_total", scalar_value=np.mean(loss_train), global_step=self._global_step
                    )
                    self._writer.add_scalar(
                        "train/loss_rot", scalar_value=np.mean(loss_train_rot), global_step=self._global_step
                    )
                    self._writer.add_scalar(
                        "train/loss_contrastive",
                        scalar_value=np.mean(loss_train_contrastive),
                        global_step=self._global_step,
                    )
                    self._writer.add_scalar(
                        "train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=self._global_step
                    )
                    self._writer.add_scalar(
                        "validation/loss_total", scalar_value=val_loss_mean, global_step=self._global_step
                    )
                    self._writer.add_scalar(
                        "validation/loss_rot", scalar_value=val_loss_rot_mean, global_step=self._global_step
                    )
                    self._writer.add_scalar(
                        "validation/loss_contrastive",
                        scalar_value=val_loss_contrastive_mean,
                        global_step=self._global_step,
                    )
                    self._writer.add_scalar(
                        "validation/loss_recon", scalar_value=val_loss_recon_mean, global_step=self._global_step
                    )

                    self._writer.add_image("validation/x1_gt", img_list[0], self._global_step, dataformats="HW")
                    self._writer.add_image("validation/x1_aug", img_list[1], self._global_step, dataformats="HW")
                    self._writer.add_image("validation/x1_recon", img_list[2], self._global_step, dataformats="HW")

                    if val_loss_recon_mean < val_best:
                        val_best = val_loss_recon_mean
                        checkpoint = {
                            "global_step": self._global_step,
                            "epoch": count_epoch,
                            "state_dict": self._model.state_dict(),
                            "optimizer": self._optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(self._args.logdir, "model_bestValRMSE.pt"))
                        print(
                            f"[{self._args.rank}] "
                            + "train: Model was saved! "
                            + f"Best Recon. Val Loss {val_best:.4f} "
                            + f"Recon. Val Loss {val_loss_recon_mean:.4f}"
                        )
                    else:
                        print(
                            f"[{self._args.rank}] "
                            + "train: Model was not saved! "
                            + f"Best Recon. Val Loss {val_best:.4f} "
                            + f"Recon. Val Loss {val_loss_recon_mean:.4f}"
                        )
        return loss, val_best

    def _validation(self, count_epoch: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[float]]:
        self._model.eval()
        loss_val = []
        loss_val_recon = []
        loss_val_rot = []
        loss_val_contrastive = []

        with torch.no_grad():
            for step, batch in enumerate(self._test_loader):
                val_inputs = batch["image"].to(self._args.device)
                x1, rot1 = rot_rand(self._args, val_inputs)
                x2, rot2 = rot_rand(self._args, val_inputs)
                x1_augment = aug_rand(self._args, x1)
                x2_augment = aug_rand(self._args, x2)
                with autocast(enabled=self._args.amp):
                    rot1_p, contrastive1_p, rec_x1 = self._model(x1_augment)
                    rot2_p, contrastive2_p, rec_x2 = self._model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)
                    loss, (rot_loss, contrast_loss, recon_loss) = self._loss_function(
                        rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs
                    )

                loss_val.append(loss)
                loss_val_rot.append(rot_loss)
                loss_val_contrastive.append(contrast_loss)
                loss_val_recon.append(recon_loss)

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
                if self._args.rank == 0:
                    print(
                        f"[{self._args.rank}] "
                        + "validation: "
                        + f"epoch {count_epoch}/{self._args.epochs - 1}, "
                        + f"global_step {self._global_step}/{self._args.num_steps - 1}, "
                        + f"Validation step {step}/{len(self._test_loader)}, "
                    )

        if self._args.device != torch.device("cpu"):
            torch.cuda.synchronize(self._args.device)

        loss_val_mean = torch.sum(torch.stack(loss_val), dim=0)
        loss_val_rot_mean = torch.sum(torch.stack(loss_val_rot), dim=0)
        loss_val_contrastive_mean = torch.sum(torch.stack(loss_val_contrastive), dim=0)
        loss_val_recon_mean = torch.sum(torch.stack(loss_val_recon), dim=0)

        dist.all_reduce(loss_val_mean, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(loss_val_rot_mean, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(loss_val_contrastive_mean, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(loss_val_recon_mean, op=torch.distributed.ReduceOp.SUM)

        loss_val_mean = loss_val_mean / self._args.val_ds_len
        loss_val_rot_mean = loss_val_rot_mean / self._args.val_ds_len
        loss_val_contrastive_mean = loss_val_contrastive_mean / self._args.val_ds_len
        loss_val_recon_mean = loss_val_recon_mean / self._args.val_ds_len

        loss_val_mean = loss_val_mean.item()
        loss_val_rot_mean = loss_val_rot_mean.item()
        loss_val_contrastive_mean = loss_val_contrastive_mean.item()
        loss_val_recon_mean = loss_val_recon_mean.item()

        if self._args.rank == 0:
            print(
                f"Validation loss: {loss_val_mean:.4f}, R:{loss_val_rot_mean:.4f}, C:{loss_val_contrastive_mean:.4f}, R:{loss_val_recon_mean:.4f}"
            )

        return loss_val_mean, loss_val_rot_mean, loss_val_contrastive_mean, loss_val_recon_mean, img_list


def main():
    args = parse_args()
    args.num_swin_blocks = [2, 2, args.num_swin_block, 2]
    args.num_heads_per_stage = [args.num_heads_first_stage * (2**x) for x in range(4)]
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    if args.distributed:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        env_dict = {key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=10))
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = torch.device(f"cuda:{args.local_rank}")
    else:
        print(f"[{os.getpid()}] single-GPU training")
        args.world_size = 1
        args.rank = 0
        args.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    assert args.rank >= 0

    torch.cuda.set_device(args.device)
    print(f"[{args.rank}] current gpu: {torch.cuda.current_device()}")
    args.logdir = os.path.join(
        args.logdir,
        f"run_{os.environ['SLURM_JOB_NAME']}_GPU{args.world_size:03d}_D{args.num_swin_block}_H{args.num_heads_first_stage}_"
        + args.jobID
        + "_"
        + datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
    )

    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        print(f"[{args.rank}] " + f"Writing Tensorboard logs to {args.logdir}")
        trainer = TrainerAndValidator(args, SummaryWriter(log_dir=args.logdir))
    else:
        trainer = TrainerAndValidator(args)
    checkpoint, best_val = trainer.train_all_steps()

    if args.rank == 0:
        print(f"[{args.rank}] " + f"Training Finished! Best val: {best_val}")
        torch.save(checkpoint, os.path.join(args.logdir, "model_final_epoch.pt"))
        print(f"[{args.rank}] " + "Saved model_final_epoch.pt")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
