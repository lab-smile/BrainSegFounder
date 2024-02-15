import time
from datetime import timedelta, datetime
from typing import Optional

import numpy as np
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from ATLAS_Stage2_duplicated_channels import parse_args, Args
import torch
import os
import torch.distributed as dist

from models.AtlasModel import AtlasModel
from optimizers.lr_scheduler import WarmupCosineSchedule
from utils.data_utils import get_T1T2_dataloaders


class Trainer:
    def __init__(self, args_: Args, writer: Optional[SummaryWriter] = None, model: Optional[torch.nn.Module] = None):
        self.args = args_
        self.writer = writer
        if model is not None:
            self.model = SwinUNETR(img_size=(args_.roi_x, args_.roi_y, args_.roi_z),
                                   in_channels=args_.in_channels,
                                   out_channels=args_.out_channels,
                                   feature_size=args_.feature_size,
                                   drop_rate=0.0,
                                   attn_drop_rate=0.0,
                                   dropout_path_rate=0.0,
                                   use_checkpoint=True).to(args_.device)
        else:
            self.model = AtlasModel(args_).to(args_.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args_.lr, weight_decay=args_.decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=args_.epochs)



        # TODO: reimplmement these optimizers and schedulers
        # if self.args.opt == "adam":
        #     self._optimizer = optim.Adam(
        #         params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay
        #     )
        # elif self.args.opt == "adamw":
        #     self._optimizer = optim.AdamW(
        #         params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay
        #     )
        # elif self.args.opt == "sgd":
        #     self._optimizer = optim.SGD(
        #         params=self.model.parameters(),
        #         lr=self.args.lr,
        #         momentum=self.args.momentum,
        #         weight_decay=self.args.decay,
        #     )
        # else:
        #     raise ValueError(f"Unknown value for argument --opt: {self.args.opt}")

        if self.args.resume is not None:
            try:
                model_dict = torch.load(self.args.resume)
                state_dict = model_dict["state_dict"]
                # fix potential differences in state dict keys from pre-training to fine-tuning
                if "module." in list(state_dict.keys())[0]:
                    if self.args.rank == 0:
                        print(f"[{self.args.rank}] " + "Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)
                # We now load model weights, setting param `strict` to ignore non-matching key, i.e.:
                # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
                # the decoder weights untouched (CNN UNet decoder).

                self.model.load_state_dict(state_dict, strict=False)
                if "epoch" in model_dict:
                    self.model.epoch = model_dict["epoch"]
                if "optimizer" in model_dict:
                    self.model.optimizer = model_dict["optimizer"]
                if self.args.rank == 0:
                    print(f"[{self.args.rank}] " + "Using pretrained self-supervised Swin UNETR backbone weights!")
                    print(f'Original keys: {state_dict.keys()} \n | Current keys: {self.model.state_dict().keys()}')
            except ValueError:
                raise ValueError("Self-supervised pre-trained weights not available for" + str(self.args.resume))

        # TODO: reimplment as necessary
        # if self.args.lrdecay:
        #     if self.args.lr_schedule == "warmup_cosine":
        #         self._scheduler = WarmupCosineSchedule(
        #             self._optimizer, warmup_steps=self.args.warmup_steps, t_total=self.args.num_steps
        #         )
        #     elif self.args.lr_schedule == "poly":
        #         self._scheduler = torch.optim.lr_scheduler.LambdaLR(
        #             self._optimizer, lr_lambda=lambda epoch: (1 - float(epoch) / float(self.args.epochs)) ** 0.9
        #         )
        # else:
        #     self._scheduler = None

        self.loss_function = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        if self.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank])
        self._train_loader, self._test_loader = get_T1T2_dataloaders(self.args)

    def train_epoch(self, args_: Args, epoch):
        self.model.train()
        loader = self._train_loader
        start_time = time.time()
        train_loss = []
        for idx, batched_data in enumerate(loader):
            data, target = batched_data['image'].to(args_.device), batched_data['label'].to(args_.device)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.loss_function(logits, target)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
        delta_t = time.time() - start_time
        average = np.mean(train_loss)
        print(f'[{args_.device}]: Training | {epoch} average Dice {1 - average} | {delta_t}')
        if self.writer is not None:
            self.writer.add_scalar('training loss', np.mean(train_loss), epoch, delta_t)
        return average

    def validate_epoch(self, args_, epoch):
        val_loss = []
        start_time = time.time()
        loader = self._test_loader
        with torch.no_grad:
            for idx, batched_data in enumerate(loader):
                data, target = batched_data['image'].to(args_.device), batched_data['label'].to(args_.device)
                logits = self.model(data)
                loss = self.loss_function(logits, target)
                val_loss.append(loss.item())

        delta_t = time.time() - start_time
        print(f'[{args_.device}]: Validation | {epoch} average Dice {1 - np.mean(val_loss)} | {delta_t} ')
        if self.writer is not None:
            self.writer.add_scalar('validation loss', np.mean(val_loss), epoch, delta_t)

        return np.mean(val_loss)

    def train(self, args_, num_epochs, val_every=5):
        best_loss = 1e8
        train_dices = []
        val_dices = []
        for epoch in range(num_epochs):
            train_dice = self.train_epoch(args_, epoch)
            if ((epoch + 1) % val_every) == 0:
                val_dice = self.validate_epoch(args_, epoch)
                val_dices.append(val_dice)
                if val_dice < best_loss:
                    best_loss = val_dice
            train_dices.append(train_dice)

        return self.model, best_loss


if __name__ == '__main__':
    args = parse_args()

    args.num_swin_blocks = [2, 2, args.num_swin_block, 2]
    args.num_heads_per_stage = [args.num_heads_first_stage * (2 ** x) for x in range(4)]
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
        trainer = Trainer(args, SummaryWriter(log_dir=args.logdir))
    else:
        trainer = Trainer(args)

    checkpoint, best_val = trainer.train(args, args.epochs)

    if args.rank == 0:
        print(f"[{args.rank}] " + f"Training Finished! Best val: {best_val}")
        torch.save(checkpoint, os.path.join(args.logdir, "model_final_epoch.pt"))
        print(f"[{args.rank}] " + "Saved model_final_epoch.pt")

    if args.distributed:
        dist.destroy_process_group()
