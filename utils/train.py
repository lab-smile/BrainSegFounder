import os

from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: callable,
                 gpu_id: int) -> None:

        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        self.loss_fn = loss_fn
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets) -> float:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch) -> float:
        batch_size = len(next(iter(self.train_data))[0])
        print(f"GPU{self.gpu_id} is running epoch {epoch} with batch size {batch_size}")
        self.train_data.sampler.set_epoch(epoch)
        running_loss = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            running_loss += self._run_batch(source, targets)
        print(f"GPU{self.gpu_id} trained {len(self.train_data)} images")
        return running_loss / len(self.train_data)

    def _val_epoch(self, epoch):
        batch_size = len(next(iter(self.val_data))[0])
        print(f"GPU{self.gpu_id} is validating epoch {epoch} with batch size {batch_size}")
        self.val_data.sampler.set_epoch(epoch)
        running_loss = 0
        for source, targets in self.val_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            running_loss += self._run_batch(source, targets)
        print(f"GPU{self.gpu_id} validated {len(self.val_data)} images")
        return running_loss / len(self.val_data)

    def _save_checkpoint(self, epoch: int, path: str):
        weights = self.model.module.state_dict()
        path = path + '_' + str(epoch) + '.ckpt'
        torch.save(weights, path)

    def train(self, max_epochs: int, writer: SummaryWriter):
        if self.gpu_id != 0:
            writer = None
        ckpt_dir = '/blue/ruogu.fang/cox.j/GatorBrain/pretrained_weights/GB_Pretrained'
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        best_val_loss = 1e10
        for epoch in tqdm(range(max_epochs)):
            self.model.train = True
            train_loss = self._run_epoch(epoch)
            if self.gpu_id == 0:
                self.model.train = False
                val_loss = self._val_epoch(epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Loss/train', train_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f'GPU{self.gpu_id} found new best val loss: {val_loss}')
                    self._save_checkpoint(epoch, ckpt_dir)


# def setup(rank: int, world_size: int):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = ''
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#     pass
#
#
# def cleanup() -> None:
#     dist.destroy_process_group()
#
#
# def distribute(rank: int,
#                world_size: int,
#                function: str,
#                model: torch.nn.Module,
#                loader: DataLoader,
#                optim: torch.optim.Optimizer,
#                scheduler: torch.optim.lr_scheduler,
#                loss_fn: callable) -> None:
#     if function not in ["train", "validation"]:
#         raise ValueError(f'Must call distribute with either train or validation, got {function}')
#
#     print(f'Running DDP {function} on GPU {rank}')
#     setup(rank, world_size)
#     model = model().to(rank)
#     ddp_model = ddp(model, device_ids=[rank])
#     if function == 'train':
#         train_epoch(ddp_model, loader, optim, scheduler, loss_fn)
#     else:
#         validate_epoch(ddp_model, loader, loss_fn)
#     cleanup()
#
#
# def train_epoch(model: torch.nn.Module,
#                 loader: DataLoader,
#                 optim: torch.optim.Optimizer,
#                 scheduler: torch.optim.lr_scheduler,
#                 loss_fn: callable):
#     running_loss = 0.
#     for i, images in enumerate(tqdm(loader)):
#         original_images, tf_images = images  # TODO: hacky, look up proper unpacking with enumerate + tuple
#         original_images, tf_images = original_images.cuda(), tf_images.cuda()
#         optim.zero_grad()
#         pred = model(tf_images)
#         loss = loss_fn(pred, original_images)
#         loss.backward()
#         optim.step()
#         running_loss += loss.item()
#     scheduler.step()
#     return running_loss / len(loader)
#
#
# def validate_epoch(model: torch.nn.Module,
#                    loader: DataLoader,
#                    loss_fn: callable):
#     running_loss = 0
#     for i, images in enumerate(tqdm(loader)):
#         val_images, val_transform = images
#         val_images, val_transform = val_images.cuda(), val_transform.cuda()
#         pred = model(val_transform)
#         loss = loss_fn(pred, val_images)
#         running_loss += loss
#     avg_val_loss = running_loss / len(loader)
#     return avg_val_loss
