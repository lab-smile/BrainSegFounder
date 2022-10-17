from torch.utils.data import Dataset
from torch.nn.functional import interpolate
import nibabel as nib
import numpy as np
import torch
from pathlib import Path


class GatorBrainDataset(Dataset):
    def __init__(self, data_dir: str | Path,
                 f_type: str,
                 subjects: list,
                 size: list = None,
                 transform: callable = None):
        super(GatorBrainDataset, self).__init__()
        self.data_dir = data_dir
        self.f_type = f_type
        self.num_subjects = len(subjects)
        self.subjects = subjects
        self.size = size
        self.transform = transform

    def __getitem__(self, key: int):
        img_path = self.data_dir / self.subjects[key]
        nib_img = nib.load(img_path)
        img = torch.as_tensor(np.array(nib_img.dataobj))

        if self.size is not None:
            img = interpolate(img.unsqueeze(0).unsqueeze(0), size=self.size, mode='nearest')

        tf_img = img
        if self.transform:
            img, tf_img = self.transform(img)

        # Save every 5000th image for visualization
        if key % 5000 == 0:
            ni_img = nib.Nifti1Image(img.numpy(), nib_img.affine)
            nib.save(ni_img, self.data_dir / 'example_images' / str(key) + "nii.gz")
        return img, tf_img

    def __len__(self):
        return self.num_subjects
