from configparser import ConfigParser, BasicInterpolation
from pathlib import Path
from monai.data import Dataset, DataLoader,ImageDataset
import nibabel as nib
import torch


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

#from utils.data import get_transforms
import pandas as pd
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_train(directory):
    csv_path = os.path.join(directory, "trainA_labels.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(directory, "trainA", filename) for filename in df['id']]
    labels = df['label'].tolist()

    return image_paths, labels
    
def get_test(directory):
    csv_path = os.path.join(directory, "testA_labels.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(directory, "testA", filename) for filename in df['id']]
    labels = df['label'].tolist()
    subjects = df['id'].str.split('M', expand=True)[1].tolist()
    return image_paths, labels, subjects
# Setup
cfg = ConfigParser(interpolation=BasicInterpolation())
cfg.read('config.ini')
general_cfg = cfg['general']
data_cfg = cfg['data']
hyperparameters = cfg['hyperparameters']
log_cfg = cfg['logging']
images, labels = get_train('/red/ruogu.fang/jbroce/ADNI_3D/')
test_images, test_labels = get_test('/red/ruogu.fang/jbroce/ADNI_3D/')
labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
test_labels = torch.nn.functional.one_hot(torch.as_tensor(test_labels)).float()
#save_path = log_cfg.get('model_path')
#log_path = log_cfg.get('log_path')
#if not os.path.exists(save_path):
#    os.mkdir(save_path)
#if not os.path.exists(log_path):
 #   os.mkdir(log_path)

# Locate images and ensure validity
#train_dir = Path(data_cfg.get('train_dir'))
#val_dir = Path(data_cfg.get('val_dir'))
preprocessing_type = {
    'T1_brain_to_MNI': 't1',
    'T1_contrast_weighted': 't1ce',
    'T2_brain': 't2',
    'T2_flair': 'flair'
}  # TODO: Put the actual names in for the other types (only T1_brain is right atm)
#try:
 #   ending = preprocessing_type[general_cfg.get('img_type')] + '.nii.gz'
#e#xcept KeyError:
#    print('Unable to parse preprocessing type, defaulting to T1 raw MRI')
 #   ending = 't1.nii.gz'
sizes = (hyperparameters.getint('pretrained_roi_x'),  # TODO: doublecheck this
         hyperparameters.getint('pretrained_roi_y'),
         hyperparameters.getint('pretrained_roi_z'))
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),Resize((sizes)), RandRotate90()])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(sizes)])

train_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=test_images, labels=test_labels, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle = False,num_workers=4, pin_memory=pin_memory)
    





def run_training(model,
    train_loader,
    val_loader,
    optimizer,
    loss_function,
    max_epochs):
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    return best_metric

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
# Load other hyperparameters
if hyperparameters.get('loss').lower() == 'dice':
    loss = DiceLoss(to_onehot_y=False,
                    sigmoid=True)
elif hyperparameters.get('loss').lower() == 'bce':
    loss = torch.nn.BCEWithLogitsLoss()
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

accuracy = run_training(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       loss,
                       5)

print(f'Fine-tuning complete: best validation accuracy {accuracy}')



