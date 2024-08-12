#!/bin/bash
#SBATCH --job-name=finetune_T1T2_S_GPU064_D18_H3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:2
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com

module load singularity
path_to_json_list="/blue/ruogu.fang/pliu1/GatorBrain/Downstream/BRATS21/jsons/brats21_folds_2modalities.json"
path_to_data_dir="/red/ruogu.fang/brats/"
path_to_checkpoint_dir="model_bestValRMSE.pt"
path_to_pretrained_dir="/red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_S_GPU064_D18_H3_07-03-2023-12:09:54-1535674/"
fold=0
depths="2 2 18 2"  # Adjust these values as needed
num_heads="3 6 12 24" # Adjust these values as needed


singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python main_FinetuningSwinUNETR.py \
--json_list=$path_to_json_list --distributed --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=2 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size 2

