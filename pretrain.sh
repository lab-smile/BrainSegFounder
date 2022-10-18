cd ~/GBT/GatorBrain || exit 1
pwd; hostname; date

module load singularity

singularity exec --nv /blue/ruogu.fang/cox.j/monaicore.latest python -c "import torch; print(torch.cuda.is_available())"
singularity exec --nv --bind /red /blue/ruogu.fang/cox.j/monaicore.latest python GatorBrain.py

