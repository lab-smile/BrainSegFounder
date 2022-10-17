cd ~/GatorBrain || exit 1
pwd; hostname; date
module load conda
conda activate /blue/ruogu.fang/cox.j/conda/envs/GatorBrain
python GatorBrain.py
