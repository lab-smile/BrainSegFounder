from torch.utils.data import Dataset
from torch.nn.functional import interpolate
import nibabel as nib
import numpy as np
import torch
from pathlib import Path


class GatorBrainDataset(Dataset):
    def __init__(self, data_dir: str | Path, f_type: str, subjects: list, size: list = None):
        super(GatorBrainDataset, self).__init__()
        self.data_dir = data_dir
        self.f_type = f_type
        self.num_subjects = len(subjects)
        self.subjects = subjects
        self.size = size

    def __getitem__(self, key: int):
        img_path = self.data_dir / self.subjects[key]
        img = torch.as_tensor(np.array(nib.load(img_path).dataobj))
        if self.size is not None:
            img = interpolate(img.unsqueeze(0).unsqueeze(0), size=self.size, mode='nearest')
        return img

    def __len__(self):
        return self.num_subjects
