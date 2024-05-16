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

cd ..
pwd;date;

module load singularity

container_path="/blue/ruogu.fang/cox.j/GatorBrainConatiner_1.3.0"
pretrained_model="/red/ruogu.fang/yyang/BraTs21_pretrain/pretrained_ssl/T1_M_model_bestValRMSE.pt"

singularty exec --bind /red --nv ${container_path} python pretrain.py -b 1 -d data/ -p ${pretrained_model} \
  -w 4 --distributed --lr_decay --max_grad_norm 5.0 --amp --url "tcp://127.0.0.1:24734"