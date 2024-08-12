#!/bin/bash
#SBATCH --job-name=Jan15_FinetuningSWINUNETR_Tiny4channels_2Stage_UKB40k_BTRATS- 8GPUs_Fold3_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
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
fold=3
max_epochs=800
#pretrained model
path_to_pretrained_dir="/blue/ruogu.fang/pliu1/SwinUNETR_pretrain_2channel/runs/run_Dec30_Stage2Pretrain_TargetBRATS_SizeTini_T1T2_Fold0_GPU002_D6_H3_01-04-2024-14:21:02/"
path_to_checkpoint_dir="model_bestValRMSE.pt"
depths="2 2 6 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed
feature_size=48
#training
batch_size=8
optim_lr=8e-4
logdir='Jan15_FinetuningSWINUNETR_Tiny4channels_2Stage_UKB40k_8GPUs_BTRATS_Fold3_'
#add timestamp to logdir


ls
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=20 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=$feature_size --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir --max_epochs=$max_epochs