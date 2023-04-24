#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:8
#SBATCH --time=1-00:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cox.j@ufl.edu

pwd;date

module load singularity

singularity exec --bind /red --nv /blue/ruogu.fang/cox.j/monaicore1.0.1 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 t1+t2.py --batch_size=1 --num_steps=10000 --lrdecay --eval_num=10000 --lr=1e-6 --decay=0.1 --logdir="./logs" 
