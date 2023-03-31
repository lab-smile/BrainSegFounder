#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:5
#SBATCH --time=01:10:00
#SBATCH --output=/blue/ruogu.fang/jbroce/GatorBrain/logs/%x.%j.out

pwd; hostname; date

module load singularity
module load git




singularity exec --writable /blue/ruogu.fang/jbroce/monaicore0.9.1 pip3 install --upgrade numpy

singularity exec --nv /blue/ruogu.fang/jbroce/monaicore0.9.1 python3 -c "import torch; print('Found GPU') if torch.cuda.is_available() else print('Could not find GPU'); import sys; print(f'Python version {sys.version}'); print(torch.__version__)"
singularity exec --nv --bind /red /blue/ruogu.fang/jbroce/monaicore0.9.1 python3 ./swin.py

