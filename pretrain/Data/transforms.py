"""
pytorch and MONAI implementation of the transforms found in ModelsGenesis paper by Zhou, Sodha, et al. For examples of
how they work see the transforms.ipynb notebook file.
"""

import monai
import numpy as np  # TODO: change to cupy as cp for HPG
import torch
import random
from random import randint
from enum import Enum


class Cutout(Enum):
    NONE = 0
    OUTER = 1
    INNER = 2


class ModelsGenesisTransformer:
    def __init__(self, window_size: int,
                 probabilities: dict,
                 seed: int = None):

        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size, window_size)
        self.nlt_probability = probabilities['nonlinear']
        self.shuffle_probability = probabilities['shuffling']
        self.cutout_probability = probabilities['cutout']
        self.seed = seed

    def __call__(self, *args, **kwargs):
        pass

    def decide_transforms(self, num_subvolumes: int, iter_number: int) -> list:
        if self.seed is not None:
            random.seed(self.seed + iter_number)
        transform_list = []
        for i in range(num_subvolumes):
            random_value = random.random()
            if random_value < self.cutout_probability:
                cutout = Cutout.OUTER if random.random() <= 0.5 else Cutout.INNER
            else:
                cutout = Cutout.NONE
            transform_list.append(
                [random_value < probability for probability in [self.nlt_probability, self.shuffle_probability]] + [
                    cutout])
        return transform_list

    @staticmethod
    def nonlinear_transformation(subvolume: torch.Tensor):
        pass

    @staticmethod
    def local_shuffling(subvolume: torch.Tensor):
        pass

    @staticmethod
    def outer_cutout(subvolume: torch.Tensor):
        pass

    @staticmethod
    def inner_cutout(subvolume: torch.Tensor):
        pass


class SwinUNETRTransforms:
    def __init__(self):
        pass

    @staticmethod
    def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
        c, h, w, z = x.size()
        n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
        mx_blk_height = int(h * max_block_sz)
        mx_blk_width = int(w * max_block_sz)
        mx_blk_slices = int(z * max_block_sz)
        tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
        total_pix = 0
        while total_pix < n_drop_pix:
            rnd_r = randint(0, h - tolr[0])
            rnd_c = randint(0, w - tolr[1])
            rnd_s = randint(0, z - tolr[2])
            rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
            rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
            rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
            if x_rep is None:
                x_uninitialized = torch.empty(
                    (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
                ).normal_()
                x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                        torch.max(x_uninitialized) - torch.min(x_uninitialized)
                )
                x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
            else:
                x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
            total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
        return x

    @staticmethod
    def rot_rand(args, x_s):
        img_n = x_s.size()[0]
        x_aug = x_s.detach().clone()
        device = torch.device(f"cuda:{args.local_rank}")
        x_rot = torch.zeros(img_n).long().to(device)
        for i in range(img_n):
            x = x_s[i]
            orientation = np.random.randint(0, 4)
            if orientation == 0:
                pass
            elif orientation == 1:
                x = x.rot90(1, (2, 3))
            elif orientation == 2:
                x = x.rot90(2, (2, 3))
            elif orientation == 3:
                x = x.rot90(3, (2, 3))
            x_aug[i] = x
            x_rot[i] = orientation
        return x_aug, x_rot

    def aug_rand(self, args, samples):
        img_n = samples.size()[0]
        x_aug = samples.detach().clone()
        for i in range(img_n):
            x_aug[i] = self.patch_rand_drop(args, x_aug[i])
            idx_rnd = randint(0, img_n-1) 
            if idx_rnd != i:
                x_aug[i] = self.patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
        return x_aug
