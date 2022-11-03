#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out

pwd; hostname; date

module load singularity
module load git

GB_PATH=~/GatorBrain
cd $GB_PATH

singularity exec --writable /blue/ruogu.fang/cox.j/monaicore0.9.1 pip3 install --upgrade numpy

singularity exec --nv /blue/ruogu.fang/cox.j/monaicore0.9.1 python3 -c "import torch; print('Found GPU') if torch.cuda.is_available() else print('Could not find GPU'); import sys; print(f'Python version {sys.version}'); print(torch.__version__)"
singularity exec --nv --bind /red /blue/ruogu.fang/cox.j/monaicore0.9.1 python3 ./GatorBrain.py

