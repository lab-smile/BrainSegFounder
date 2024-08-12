#!/bin/bash
#SBATCH --job-name=Dec29_FinetuningSWINUNETR_M4channels_2Stage_UKB40k_BTRATS-_MultipleGPUs_Fold1
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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com

module load singularity

#data
path_to_json_list="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_folds.json"
path_to_data_dir="/red/ruogu.fang/brats/"
fold=1

#pretrained model
path_to_pretrained_dir="/blue/ruogu.fang/pliu1/SwinUNETR_pretrain_2channel/runs/run_Dec28_Stage2Pretrain_TargetBRATS_Msize_T1T2_Fold1_GPU002_D2_H3_01-02-2024-14:20:11/"
path_to_checkpoint_dir="model_bestValRMSE_65k.pt"
depths="2 2 2 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed

#training
batch_size=2
optim_lr=1e-4
logdir='Dec29_FinetuningSWINUNETR_M4channels_2Stage_UKB40k_BTRATS-_MultipleGPUs_Fold1_'
#add timestamp to logdir
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir