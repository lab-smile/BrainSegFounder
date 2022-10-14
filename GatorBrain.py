import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import pathlib
from GatorBrainDataset import GatorBrainDataset
from utils.tf import init_transform
from utils.train import train_epoch, validate_epoch
from torchinfo import summary
from torch.utils.data import DataLoader
import os
import numpy
import nibabel as nib
from configparser import ConfigParser, BasicInterpolation
from monai.networks.nets.swin_unetr import SwinUNETR


# Setup
cfg = ConfigParser(interpolation=BasicInterpolation())
cfg.read('config.ini')
general_cfg = cfg['general']
data_cfg = cfg['data']
training_cfg = cfg['pretraining']
tf_cfg = cfg['transformations']
log_cfg = cfg['logging']
for key, value in log_cfg.items():
    if key.endswith('path') and not os.path.exists(value):
        os.mkdir(value)
model_name = general_cfg['model_name']
writer = SummaryWriter(log_cfg['log_path'])

# Train/Valid split
data_dir = pathlib.Path(data_cfg['dir'])
f_type = general_cfg['img_type'] + '.nii.gz'
train_fraction = training_cfg.getfloat('train_fraction')
subjects = list(filter(lambda s_path: (data_dir / s_path).is_file(),
                       [subject / f_type for subject in data_dir.iterdir()]))

train, valid = train_test_split(subjects, train_size=train_fraction)

if general_cfg.getboolean('small_train') and len(train) > 1000:
    train = train[0:1000]
    valid = valid[0:round(1000*(1-train_fraction))]

sample_img = list(torch.as_tensor(numpy.array(nib.load(data_dir/subjects[0]).dataobj)).size())
sizes = [32 * (i//32 + 1) for i in sample_img]
train_set = GatorBrainDataset(data_dir, f_type, train, sizes)
valid_set = GatorBrainDataset(data_dir, f_type, valid, sizes)
print(f'Training on {len(train_set)} images and validating with {len(valid_set)} images.')


# Format Model
batch_size = training_cfg.getint('batch_size')
device = torch.device('cuda')
model = SwinUNETR(img_size=sizes,
                  in_channels=1,
                  out_channels=1)
model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
model.to(device)

print(f'Using {torch.cuda.device_count()} CUDA device(s)')
print('Loaded model: ')
in_size = [batch_size, 1] + sizes
summary(model, input_size=in_size)


# Pretrain Model
print('Preparing model for training...')
initial_epoch, max_epochs = 0, training_cfg.getint('nb_epochs')
optimizer, lr = training_cfg['optimizer'], training_cfg.getfloat('learning_rate')
patience = training_cfg.getint('patience')
criterion = torch.nn.MSELoss()

if optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr)
elif optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0, nesterov=False)
else:
    raise ValueError(f'Selected optimizer must be one of ["adam", "sgd"]: got {optimizer}')

workers = training_cfg.getint('workers')

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(patience * 0.8), gamma=0.5)
training_loader = DataLoader(train_set, batch_size=batch_size,
                             shuffle=True, num_workers=workers)
validation_loader = DataLoader(valid_set, batch_size=batch_size,
                               shuffle=True, num_workers=workers)

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

best_loss = 100000
model_path = None
num_epochs_since_improve = 0
patience = training_cfg.getint('patience')
slice_transform = init_transform(tf_cfg)
print("Beginning pretraining...")

for epoch in range(initial_epoch, max_epochs):
    print(f'Training epoch {epoch + 1}')
    model.train(True)
    avg_train_loss = train_epoch(model, training_loader, optimizer, scheduler, criterion, tf_cfg, slice_transform)
    writer.add_scalar('train_loss', avg_train_loss, epoch)
    model.train(False)
    with torch.no_grad():
        print(f"Validating epoch {epoch + 1}")
        avg_val_loss = validate_epoch(model, validation_loader, criterion, tf_cfg, slice_transform)
        print(f'LOSS validation - {avg_val_loss}, train - {avg_train_loss}')
        writer.add_scalar('val_loss', avg_val_loss, epoch)
    if avg_val_loss < best_loss:
        num_epochs_since_improve = 0
        print("New best loss.")
        best_loss = avg_val_loss
        model_path = f"{log_cfg['model_path']}/{model_name}_{epoch}"
        torch.save(model.state_dict(), model_path)
    else:
        num_epochs_since_improve += 1
    if num_epochs_since_improve == patience:
        print("I'm at my wit's end! (Patience reached - stopping)")
        break

print(f"Pretraining complete. Best model saved in {model_path}")
