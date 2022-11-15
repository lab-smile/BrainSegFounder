from configparser import ConfigParser, BasicInterpolation
from pathlib import Path
from monai.data import Dataset, DataLoader
import nibabel as nib
import torch
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR

from model import GBPretrainedModel
import os
import numpy as np
from utils.data import get_transforms

# Setup
cfg = ConfigParser(interpolation=BasicInterpolation())
cfg.read('config.ini')
general_cfg = cfg['general']
data_cfg = cfg['data']
hyperparameters = cfg['hyperparameters']
log_cfg = cfg['logging']

save_path = log_cfg.get('model_path')
log_path = log_cfg.get('log_path')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

# Locate images and ensure validity
train_dir = Path(data_cfg.get('train_dir'))
val_dir = Path(data_cfg.get('val_dir'))
preprocessing_type = {
    'T1_brain_to_MNI': 't1',
    'T1_contrast_weighted': 't1ce',
    'T2_brain': 't2',
    'T2_flair': 'flair'
}  # TODO: Put the actual names in for the other types (only T1_brain is right atm)
try:
    ending = preprocessing_type[general_cfg.get('img_type')] + '.nii.gz'
except KeyError:
    print('Unable to parse preprocessing type, defaulting to T1 raw MRI')
    ending = 't1.nii.gz'

train_list = list(filter(lambda s_path: (train_dir / s_path).is_file(),
                         [subject / (str(subject) + ending) for subject in train_dir.iterdir()]))

val_list = list(filter(lambda s_path: (val_dir / s_path).is_file(),
                       [subject / (str(subject) + ending) for subject in val_dir.iterdir()]))

sample_img = (np.array(nib.load(train_list[0]).dataobj))
roi_x, roi_y, roi_z = np.shape(sample_img)
transforms = get_transforms(roi_x, roi_y, roi_z)

# Create datasets and dataloaders
batch_size = hyperparameters.getint('per_gpu_batch_size')
workers = hyperparameters.getint('workers')

train_ds = Dataset(data=train_list, transform=transforms[0])
val_ds = Dataset(data=val_list, transform=transforms[1])

train_loader = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=workers,
                          pin_memory=True)

val_loader = DataLoader(val_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=workers,
                        pin_memory=True)

# Create Model
pretrained_in = 1
pretrained_out = 1
sizes = (hyperparameters.getint('pretrained_roi_x'),  # TODO: doublecheck this
         hyperparameters.getint('pretrained_roi_y'),
         hyperparameters.getint('pretrained_roi_z'))
pretrained_model = SwinUNETR(img_size=sizes,
                             in_channels=pretrained_in,
                             out_channels=pretrained_out)
model = GBPretrainedModel(pretrained_model,
                          in_channels=1,
                          out_channels=4,  # num classes in Brats
                          model_params=(pretrained_in, pretrained_out))
ckpt = torch.load(data_cfg.get('load_from'))

model.load_state_dict(ckpt, strict=False)
print('Successfully loaded checkpoint, beginning finetune...')

# Load other hyperparameters
if hyperparameters.get('loss').lower() == 'dice':
    loss = DiceLoss(to_onehot_y=False,
                    sigmoid=True)
else:
    raise ValueError(f'Only dice loss implemented, got {hyperparameters.get("loss").lower()}')

if hyperparameters.get('optimizer').lower() == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hyperparameters.getfloat('learning_rate'),
                                  weight_decay=hyperparameters.getfloat('reg_weight'))
else:
    raise ValueError(f'Unsupported optimization procedure {hyperparameters.get("optimizer")}')

best_acc = -999
epoch = 0

semantic_classes = ['Dice_Val_TC', 'Dice_Val_WT', 'Dice_Val_ET']

accuracy = run_training()

print(f'Fine-tuning complete: best validation accuracy {accuracy}')



