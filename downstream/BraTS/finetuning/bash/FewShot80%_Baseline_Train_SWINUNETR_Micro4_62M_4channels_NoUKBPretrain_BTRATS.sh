#!/bin/bash
#SBATCH --job-name=Mar26_Fewshot80_Baseline_Pretrain_SWINUNETR_62M_4channels_NoUKBPretrain_BTRATS_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12gb
#SBATCH --partition=hpg-ai
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --exclusive
module load singularity

#data
json_list_train="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_fewshot_train_80%.json"
json_list_test="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_fewshot_fixed_test_set.json"
path_to_data_dir="/red/ruogu.fang/brats/"

depths="2 2 2 2"  # Adjust these values as needed
feature_size=48
num_heads="3 6 12 24" # Adjust these values as needed
#training
batch_size=2
optim_lr=1e-4
logdir='Mar26_Fewshot80_Baseline_Pretrain_SWINUNETR_62M_4channels_NoUKBPretrain_BTRATS_'
#add timestamp to logdir


ls
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python /blue/ruogu.fang/pliu1/BRATS21/main_BaselineSwinUNETR_4Channels_Fewshot.py \
--json_list_train=$json_list_train --json_list_test=$json_list_test \
--distributed --data_dir=$path_to_data_dir --val_every=10 --noamp --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --feature_size=$feature_size --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir