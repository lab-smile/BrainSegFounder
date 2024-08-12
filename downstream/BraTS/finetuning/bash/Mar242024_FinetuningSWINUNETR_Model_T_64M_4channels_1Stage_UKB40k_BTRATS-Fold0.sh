#!/bin/bash
#SBATCH --job-name=Mar23_FinetuningSWINUNETR_Model_T_64M_M4channels_1Stage_UKB40k_BTRATS-Fold0_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --partition=hpg-ai
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

module load singularity

#ajdust data ****************************************************
path_to_json_list="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_folds.json"
path_to_data_dir="/red/ruogu.fang/brats/"

#**************ajust the fold number***********************************
fold=0

#**************pretrained model path by looking at the ppt file **************
path_to_pretrained_dir="/red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_T_GPU064_D6_H3_07-03-2023-12:03:35-1534968/"
path_to_checkpoint_dir="model_bestValRMSE.pt"

#**************check the ppt to find the correct ones for the following **************
depths="2 2 6 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed

#**************logdir name by looking at the ppt file **************
logdir='Mar23_FinetuningSWINUNETR_Model_T_64M_M4channels_1Stage_UKB40k_BTRATS'

#training
batch_size=2
optim_lr=1e-4
#add timestamp to logdir


ls
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir