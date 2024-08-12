#!/bin/bash
#SBATCH --job-name=Apr20_Baseline_M4channels_NoUKBPretrain_BTRATS_T1_Fold1_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24gb
#SBATCH --partition=hpg-ai
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

module load singularity

#data
path_to_json_list="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_folds_T1_4copy.json"
path_to_data_dir="/red/ruogu.fang/brats/"
fold=1

depths="2 2 2 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed
#training
batch_size=2
optim_lr=1e-4
logdir='Apr20_Baseline_M4channels_NoUKBPretrain_BTRATS_T1_4copy_Fold1_'
#add timestamp to logdir


ls
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python /blue/ruogu.fang/pliu1/BRATS21/main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=5 --noamp \
--fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir