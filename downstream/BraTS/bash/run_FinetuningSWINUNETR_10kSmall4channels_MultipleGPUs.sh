#!/bin/bash
#SBATCH --job-name=finetune_T1T2_S_10k_Nov252023_4Channels_8GPUs_LearningRate1e-3_Fold0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:8
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com

module load singularity
path_to_json_list="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_folds.json"
path_to_data_dir="/red/ruogu.fang/brats/"
path_to_checkpoint_dir="T1T2_S_model_bestValRMSE.pt"
path_to_pretrained_dir="/red/ruogu.fang/yyang/monaicore_swinUNETR/pretrained_ssl_10K/"
fold=0
depths="2 2 18 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed
batch_size=1
optim_lr=1e-3

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr