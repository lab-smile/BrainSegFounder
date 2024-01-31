from os import PathLike
import os
from pathlib import Path

import monai.transforms
from monai import data, transforms
import torch
from sklearn.model_selection import train_test_split


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
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
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

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
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def split_atlas_data(data_directory: str, train_fraction: float = 0.8, seed: int = None) -> tuple:
    """Locates and splits ATLAS 2.0 Data into training and validation lists.

    :param data_directory: absolute path to the source file. For ATLAS should be /red/ruogu.fang/atlas/decrypt/ATLAS_2
    :param train_fraction: percentage of the data split into training data
    :param seed: random seed for splitting data. If None, data will be split differently each time.
    :returns: tuple of training data, validation data. Each item in the tuple is a list of dictionaries pointing to the
        with keys 'image' and 'label', similar to how the BraTS data is treated
    """

    data_directory = Path(data_directory)
    training_dir = data_directory / 'Training'
    subdirectory = 'ses-1/anat/'

    training = []
    for record_id in training_dir.iterdir():
        for subject_id in record_id.iterdir():
            if str(subject_id).endswith('.json'):
                continue
            scan_path = training_dir / record_id / subject_id / subdirectory
            label = f'{scan_path}/{os.path.basename(subject_id)}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'
            image = f'{scan_path}/{os.path.basename(subject_id)}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz'

            training.append(
                {
                    'label': label,
                    'image': image
                }
            )

    train, val = train_test_split(training, train_size=train_fraction, random_state=seed)
    return train, val


def get_loaders(data_directory: str,
                batch_size: int,
                roi: tuple = (96, 128, 128),
                train_fraction: float = 0.8,
                seed: int = None,
                n_workers: int = 1,
                distributed=False):
    """Create training and validation dataloaders for the ALTAS Dataset

    :param data_directory: absolute path to the high level directory for data. For ATLAS,
        should be /red/ruogu.fang/atlas/decrypt/ATLAS_2.
    :param batch_size: Batch size for training stage (validation size is 1)
    :param roi: Region of interest, images will be cropped to this size (x, y, z)
    :param train_fraction: percentage of the data split into training data
    :param seed: random seed for splitting data. If None, data will be split differently each time.
    :param n_workers: number of workers for each dataloader. Set to max number of CPU cores available for best results
    :param distributed: Whether or not to use multi-GPU sampler. Set to True if using multi-GPU training.
    """

    train_files, validation_files = split_atlas_data(data_directory, train_fraction, seed)
    train_transforms, validation_transforms = get_transforms(roi)

    train_dataset = data.Dataset(data=train_files, transform=train_transforms)
    train_sampler = Sampler(train_dataset) if distributed else None
    train_loader = data.DataLoader(train_dataset,
                                   num_workers=n_workers,
                                   shuffle=(not distributed),
                                   batch_size=batch_size,
                                   pin_memory=True,
                                   sampler=train_sampler)

    validation_dataset = data.Dataset(data=validation_files, transform=validation_transforms)
    validation_sampler = Sampler(validation_dataset) if distributed else None
    validation_loader = data.DataLoader(validation_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=n_workers,
                                        sampler=validation_sampler,
                                        pin_memory=True)

    return train_loader, validation_loader


def get_transforms(roi: tuple) -> tuple:
    """Creates two transforms for the data, one for training and validation. The training transform consists of
        random image augmentation steps, controlled by MONAI's global seed. If you want to change what augmentation
        is performed during this stage of pretraining, this is where you change it. For more information on the
        transforms used, see the README for this downstream Task

    :param roi: tuple of 3 integers, the desired size of the output images in x, y, z coordinates.
    :returns: tuple of monai.transform.Compose objects (training, validation).
    """
    all_keys = ['image', 'label']

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=all_keys),
            transforms.EnsureChannelFirstd(keys=all_keys),
            transforms.RepeatChanneld(keys='image', repeats=2),  # Repeat channel to match 2 channel input
            transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
            transforms.Resized(keys=all_keys, spatial_size=roi),
            transforms.ToTensord(keys=all_keys)
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=all_keys),
            transforms.RepeatChanneld(keys='image', repeats=1),
            transforms.Resized(keys=all_keys, spatial_size=roi),
            transforms.ToTensord(keys=all_keys)
        ]
    )

    return train_transform, val_transform

