#!/bin/bash
#SBATCH --job-name=finetune_T1T2_L_GPU128_D18_H6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com

module load singularity
path_to_json_list="/blue/ruogu.fang/pliu1/GatorBrain/Downstream/BraTS/jsons/brats21_folds.json"
path_to_data_dir="/red/ruogu.fang/brats/"
path_to_checkpoint_dir="model_bestValRMSE.pt"
path_to_pretrained_dir="/red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_L_GPU128_D18_H6_07-04-2023-23:09:48/"
fold=0


singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python main.py --json_list=$path_to_json_list --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=2 --spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=96
