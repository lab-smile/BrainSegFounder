import os
import json
from functools import partial

import torch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data

from finetune_args import parse_args
from trainer import trainer

# Each pretrained model needs a name here
model_hyperparameters = {
    'name': {
        'out_channels': 1,
        'batch_wise': True,
        'num_modalities': 2
    }
}


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


def get_loader(batch_size_, data_dir_, json_list, fold_, roi_, num_workers_=8):
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

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader_ = data.DataLoader(train_ds,
                                    batch_size=batch_size_,
                                    shuffle=True,
                                    num_workers=num_workers_,
                                    pin_memory=True)

    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader_ = data.DataLoader(val_ds,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers_,
                                  pin_memory=True)

    return train_loader_, val_loader_


if __name__ == '__main__':
    args = parse_args()
    output_dir = args.output
    data_dir = args.data_dir
    model_path = args.checkpoint_dir
    roi = args.roi
    batch_size = args.batch_size
    fold = args.fold
    num_epochs = args.epochs
    json_path = args.json_path
    num_workers = args.num_workers

    hyperparameters = model_hyperparameters[model_path.split('/')[-1]]
    sw_batch_size = hyperparameters['out_channels'] * (
        hyperparameters['num_modalities'] if hyperparameters['batch_wise'] else 1)

    if not torch.cuda.is_available():
        raise ValueError("CUDA enabled GPU is necessary for this code to run, sorry")  # ?
    device = torch.device('cuda')

    train_loader, val_loader = get_loader(batch_size, data_dir, json_path, fold, roi, num_workers_=num_workers)

    model = SwinUNETR(
        img_size=roi,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    ).to(device)

    torch.backends.cudnn.benchmark = True  # Optimizes our runtime a lot

    loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inference = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=0.5,  # To be honest, I am unsure what this parameter does - need to investigate more
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoint = torch.load(model_path)
    # Cover each of the three different "types" of saved models
    if 'optimizer_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dit'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch = \
        trainer(model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_func=loss,
                acc_func=dice_acc,
                scheduler=scheduler,
                max_epochs=num_epochs,
                batch_size=batch_size,
                device=device,
                model_inferer=model_inference,
                start_epoch=0,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred)
    print(f"Final model saved at {output_dir}/model.pt")
