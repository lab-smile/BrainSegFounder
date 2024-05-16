#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:2
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cox.j@ufl.edu

cd ..;
pwd;date;

module load singularity

container_path="/blue/ruogu.fang/cox.j/GatorBrainConatiner_1.3.0"
pretrained_model="/red/ruogu.fang/yyang/BraTs21_pretrain/pretrained_ssl/T1_M_model_bestValRMSE.pt"

singularity exec --bind /red --nv ${container_path} python finetune.py --checkpoint ${pretrained_model} \
  --logdir logs/finetune/ --data_dir data/ --output models/ --num_workers 1 --batch_size 2 --epochs 1000 \
  --seed 1234 --distributed --in_channels 1 --out_channels 1 --feature_size 48 --depths 2 2 2 2 --dropout_rate 0.1 \
  --amp --heads 3 6 12 24