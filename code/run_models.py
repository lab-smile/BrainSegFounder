from configparser import ConfigParser, BasicInterpolation
from pathlib import Path
from monai.data import Dataset, DataLoader,ImageDataset
import nibabel as nib
import torch
import monai
from trainer import run_training
from sklearn.model_selection import GroupKFold
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from model import GBPretrainedModel
import os
import numpy as np
import sys

#from utils.data import get_transforms
import pandas as pd
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def get_files(directory):
    csv_path = os.path.join(directory, "labels.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(directory, "images", filename) for filename in df['id']]
    labels = df['label'].tolist()
    subjects = df['id'].str.split('M', expand=True)[1].tolist()
    return image_paths, labels, subjects
# Setup

cfg = ConfigParser(interpolation=BasicInterpolation())

if len(sys.argv) > 1:
  # get the first arg after the script name
  cfg.read(sys.argv[1])
else:
  # use a default name if no arg is given
  cfg.read('config.ini')

general_cfg = cfg['general']
data_cfg = cfg['data']
hyperparameters = cfg['hyperparameters']
log_cfg = cfg['logging']
images, labels ,subjects= get_files('/red/ruogu.fang/jbroce/ADNI_3D/')
model_cfg = cfg['model']



preprocessing_type = {
    'T1_brain_to_MNI': 't1',
    'T1_contrast_weighted': 't1ce',
    'T2_brain': 't2',
    'T2_flair': 'flair'
} 
sizes = (hyperparameters.getint('pretrained_roi_x'),  
         hyperparameters.getint('pretrained_roi_y'),
         hyperparameters.getint('pretrained_roi_z'))


best_metrics = []

train_batch_size = hyperparameters.getint('train_batch_size')
validation_batch_size = hyperparameters.getint('validation_batch_size')



# Load other hyperparameters

if hyperparameters.get('loss').lower() == 'dice':
    loss = DiceLoss(to_onehot_y=False,
                    sigmoid=True)
elif hyperparameters.get('loss').lower() == 'bce':
    loss = torch.nn.BCEWithLogitsLoss()
else:
    raise ValueError(f'Only dice loss implemented, got {hyperparameters.get("loss").lower()}')



best_acc = -999
epoch = 0
group_kfold = GroupKFold(n_splits=5)
for i, (train_index, test_index) in enumerate(group_kfold.split(images, labels ,subjects)):
    if model_cfg.get('model_name') == 'densenet':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    else:
       
    # Create Model
        pretrained_in = 1
        pretrained_out = 1

        pretrained_model = SwinUNETR(img_size=sizes,
                                 in_channels=pretrained_in,
                                 out_channels=pretrained_out)
        model = GBPretrainedModel(pretrained_model,
                              in_channels=1,
                              out_channels=2,  # num classes in Brats 4 -> 2
                              model_params=(pretrained_in, pretrained_out))
        ckpt = torch.load(data_cfg.get('load_from'))

        model.load_state_dict(ckpt, strict=False)
        print('Successfully loaded checkpoint, beginning finetune...')
    model.to(device)
    if hyperparameters.get('optimizer').lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hyperparameters.getfloat('learning_rate'),
                                  weight_decay=hyperparameters.getfloat('reg_weight'))
    else:
        raise ValueError(f'Unsupported optimization procedure {hyperparameters.get("optimizer")}')
    train_labels,test_labels = np.array(labels)[train_index], np.array(labels)[test_index]
    train_labels = torch.nn.functional.one_hot(torch.as_tensor(train_labels)).float()
    test_labels = torch.nn.functional.one_hot(torch.as_tensor(test_labels)).float()
    images = np.array(images)
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),Resize((sizes)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(sizes)])

    train_ds = ImageDataset(image_files=images[train_index].tolist(), labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)

# create a validation data loader
    val_ds = ImageDataset(image_files=images[test_index].tolist(), labels=test_labels, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=validation_batch_size, shuffle = False,num_workers=4, pin_memory=pin_memory)
    
    best_metrics.append(run_training(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       loss,
                       train_ds,
                       5))

print(f'Fine-tuning complete: best validation accuracies {best_metrics}')



