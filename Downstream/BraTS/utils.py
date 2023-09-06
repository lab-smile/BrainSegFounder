import json
import os
from monai import transforms
from monai import data
import torch


def datafold_read(datalist, basedir, fold_=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold_:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model_, epoch, filename="model.pt", best_acc=0, dir_add='./models'):
    state_dict = model_.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print(f"Saving checkpoint to {filename}")


def get_loader(batch_size_, data_dir_, json_list, fold_, roi_, num_workers_=8, world_size=1, rank=0):
    data_dir_ = data_dir_
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir_, fold_=fold_)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi_[0], roi_[1], roi_[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi_[0], roi_[1], roi_[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.partition_dataset(data.Dataset(data=train_files, transform=train_transform),
                                      num_partitions=world_size)[rank]

    train_loader_ = data.DataLoader(train_ds,
                                    batch_size=batch_size_,
                                    shuffle=True,
                                    num_workers=num_workers_,
                                    pin_memory=True)

    val_ds = data.partition_dataset(data.Dataset(data=validation_files, transform=val_transform),
                                    num_partitions=world_size)[rank]
    val_loader_ = data.DataLoader(val_ds,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers_,
                                  pin_memory=True)

    return train_loader_, val_loader_
