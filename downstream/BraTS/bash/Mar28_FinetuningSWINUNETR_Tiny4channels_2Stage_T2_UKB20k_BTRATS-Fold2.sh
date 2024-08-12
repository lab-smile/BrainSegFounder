#!/bin/bash
#SBATCH --job-name=Mar29_FinetuningSWINUNETR_Tiny4channels_2Stage_T2_UKB20k_BTRATS_2GPUs_Fold2_
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

module load singularity

#data
path_to_json_list="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_folds.json"
path_to_data_dir="/red/ruogu.fang/brats/"
fold=0
max_epochs=800
#pretrained model
path_to_pretrained_dir="/blue/ruogu.fang/pliu1/SwinUNETR_pretrain_2channel/runs/run_Mar282024_Stage2Pretrain_TargetBRATS_Size_Tini_T2_UKB20K_Fold2__GPU002_D6_H3_03-28-2024-17:42:35/"
path_to_checkpoint_dir="model_bestValRMSE.pt"
depths="2 2 6 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed
feature_size=48
#training
batch_size=2
optim_lr=8e-4
logdir='Mar29_FinetuningSWINUNETR_Tiny4channels_2Stage_T2_UKB20k_BTRATS_2GPUs_Fold2_'
#add timestamp to logdir


ls
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python /blue/ruogu.fang/pliu1/BRATS21/main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=20 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=$feature_size --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir --max_epochs=$max_epochs