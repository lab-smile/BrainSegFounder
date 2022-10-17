#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out

date; hostname; pwd

module load singularity

GB_PATH=~/GBT2/GatorBrain/

singularity build --sandbox /blue/ruogu.fang/cox.j/monaicore.latest docker://projectmonai/monai:latest
singularity exec --writable /blue/ruogu.fang/cox.j/monaicore.latest pip install -r ${GB_PATH}/requirements.txt