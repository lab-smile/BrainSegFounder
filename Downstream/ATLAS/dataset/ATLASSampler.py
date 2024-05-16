import math

import torch
from .ATLASDataset import ATLASDataset
from torch.utils.data import Sampler
from typing import Optional
import numpy as np


class ATLASSampler(Sampler):
    def __init__(self, dataset: ATLASDataset, num_replicas: int = None, rank: int = None,
                 shuffle: bool = True,
                 make_even: bool = True):
        super().__init__()
        _validate_runtime(num_replicas, rank)
        num_replicas = torch.distributed.get_world_size() if num_replicas is None else num_replicas
        rank = torch.distributed.get_rank() if rank is None else rank
        indices = [i for i in range(len(dataset))]
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_replicas = num_replicas
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.valid_length = len(indices[rank:self.total_size:num_replicas])

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = [i for i in range(len(self.dataset))]
            if self.make_even:
                if len(indices) < self.total_size:
                    if self.total_size - len(indices) < len(indices):
                        indices += indices[:(self.total_size - len(indices))]
                    else:
                        extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                        indices += [indices[ids] for ids in extra_ids]
                assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


def _validate_runtime(num_replicas: Optional[int], rank: Optional[int]) -> None:
    if (num_replicas is None or rank is None) and (not torch.distributed.is_available()):
        raise RuntimeError('PyTorch distributed package is not available and needs to be.')
