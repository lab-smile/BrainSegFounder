#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=07:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cox.j@ufl.edu

date

module load conda
conda activate /blue/ruogu.fang/cox.j/conda/envs/GatorBrain/

echo "python executable at `which python`"

python /red/ruogu.fang/cox.j/SwinValidation/Pretrain/conversion.py

date
