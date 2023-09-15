import torch
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from functools import partial
from monai.data import Dataset, decollate_batch
from monai import transforms, data
import math
import numpy as np
import os
import json

from monai.transforms import Activations, AsDiscrete
from monai.utils import MetricReduction

from Downstream.BraTS.AverageMeter import AverageMeter
from Downstream.BraTS.trainer import calculate_individual_dice

test_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "label"]),
    ]
)


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

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    if fold == "all":
        fold = 0
        all_data = True
    else:
        all_data = False
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    if all_data:
        val = val + tr
    return tr, val


finetuned_models = ['batchwise10k_t1t2.pt',
                    'frozen_encoder.pt',
                    'GatorBrain_Brats_channelwise.pt',
                    'GatorBrain_T1_ONLY.pt']
model_path = './models'
data_dir = '/red/ruogu.fang/brats'
datalist_json = 'brats21_folds.json'
json_dir = './jsons'
print("CUDA:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,
).to(device)

post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

f = open('inference.csv', 'w', encoding='utf-8')
f.write('model,fold,img,tc,wt,et,avg')
for fold in [0, 1, 2, 3, 4, 'all']:
    _, validation_files = datafold_read(datalist_json, './jsons', fold=fold)
    val_ds = Dataset(data=validation_files, transform=test_transform)
    test_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    for finetuned_model in finetuned_models:
        model.load_state_dict(torch.load(f'{model_path}/{finetuned_model}'))
        model.eval()
        model.to(device)

        model_inferer_test = partial(
            sliding_window_inference,
            roi_size=[128, 128, 128],
            sw_batch_size=1,
            predictor=model,
            overlap=0.6)

        with torch.no_grad():
            for image_num, batch in enumerate(test_loader):
                run_acc = AverageMeter()
                image, target = batch["image"].to(device), batch["label"].to(device)
                num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
                logits = model_inferer_test(image)
                img_name = "BraTS2021_" + num + ".nii.gz"
                print(f"Inference on case {img_name}")
                dice_tc, dice_wt, dice_et = calculate_individual_dice(target, logits, run_acc, post_pred,
                                                                      post_sigmoid,
                                                                      dice_acc)

                avg = (dice_tc+dice_wt+dice_et)/3.0
                f.write(f'{finetuned_model},{fold},{img_name},{dice_tc},{dice_wt},{dice_et},{avg}')
        print(f"Finished inference on fold {fold} with model {finetuned_model}!")
f.close()
