import json
import math
import numpy as np
import torch

from pytorch_utils import distributed_torch_available
from transforms import get_transforms
from monai import data, transforms


def get_loader(json_file: str, task: str, fold: str, first_reviewer_only: bool, roi: list, batch_size: int):
    train_files, validation_files = datafold_read(json_file, task, fold, first_reviewer_only)
    train_transform, val_transform = get_transforms(key='train', roi=roi), get_transforms(key='val', roi=roi)

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True)

    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True)
    
    return train_loader, val_loader


def datafold_read(datalist: str, task: str, fold: str, first_reviewer_only: bool = True):
    with open(datalist) as f:
        json_data = json.load(f)

    task_files = json_data[task]
    fold_data = task_files[fold]
    train, val = fold_data['training'], fold_data['validation']
    
    if first_reviewer_only:
        train, val = remove_alternative_reviewers(train), remove_alternative_reviewers(val)
    return train, val


def remove_alternative_reviewers(dataset: list[dict]):
    for subject in dataset:
        subject['label'] = [subject['label'][0]]
    return dataset


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            distributed_torch_available()
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            distributed_torch_available()
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

