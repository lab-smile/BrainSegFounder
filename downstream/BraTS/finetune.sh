#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=finetune_batchwise10k__%Y%m%d%H%M%S.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=USER

module load singularity

singularity exec --bind /red --nv /blue/ruogu.fang/cox.j/GatorBrainContainer python finetune.py --single_gpu -d /red/ruogu.fang/brats/ -c /blue/ruogu.fang/cox.j/GatorBrain/models/batchwise10k_t1t2.pt 
