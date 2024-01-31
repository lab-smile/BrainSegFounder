#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:2
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out

cd .. 
pwd; date; time;

module load singularity

singularity exec --bind /red --nv /blue/ruogu.fang/cox.j/GatorBrainContainer_1.2.0 python main_T1T2.py --resume /red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_S_GPU064_D18_H3_07-03-2023-12:09:54-1535674/model_bestValRMSE.pt --logdir . --workdir . --in_channels=2 --roi_x 128 --roi_y 128 --roi_z 128 --batch_size=1 --T1T2_target_Brats --split_json ./jsons/atlas_2_folds_2channel.json --target_data_fold=2 --target_data_path /red/ruogu.fang/atlas/decrypt/ATLAS_2 
