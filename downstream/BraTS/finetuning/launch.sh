#!/bin/bash
#SBATCH --job-name=ExampleBratsFinetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --partition=hpg-ai
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out

module load singularity

#data
path_to_json_list="/path/to/folds.json"
path_to_data_dir="/path/to/brats/"
path_to_singularity_container="/path/to/container/"
fold=0

#pretrained model
path_to_pretrained_dir="/path/to/pretrained/models/"
path_to_checkpoint_dir="model_bestValRMSE.pt"
depths="2 2 2 2"  
num_heads="3 6 12 24" 
#training
batch_size=2
optim_lr=1e-4
logdir='log-'
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv $path_to_singularity_container python main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir
