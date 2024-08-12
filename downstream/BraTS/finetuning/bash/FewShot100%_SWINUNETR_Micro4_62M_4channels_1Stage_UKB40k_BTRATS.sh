#!/bin/bash
#SBATCH --job-name=Mar26_Fewshot100_SWINUNETR_62M_4channels_1Stage_UKB40k_BTRATS_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12gb
#SBATCH --partition=hpg-ai
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --exclusive
module load singularity

#data
json_list_train="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_fewshot_train_100%.json"
json_list_test="/blue/ruogu.fang/pliu1/BRATS21/jsons/brats21_fewshot_fixed_test_set.json"
path_to_data_dir="/red/ruogu.fang/brats/"

#pretrained model
path_to_pretrained_dir="/red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_M_GPU128_07-02-2023-10:02:30-1458786/"
path_to_checkpoint_dir="model_bestValRMSE.pt"

depths="2 2 2 2"  # Adjust these values as needed
feature_size=48
num_heads="3 6 12 24" # Adjust these values as needed
#training
batch_size=2
optim_lr=1e-4
logdir='Mar26_Fewshot100_SWINUNETR_62M_4channels_1Stage_UKB40k_BTRATS_'
#add timestamp to logdir


ls
logdir+=$(date "+%Y-%m-%d-%H:%M:%S")

singularity exec --bind /red --nv /blue/ruogu.fang/pliu1/GatorBrainContainer python /blue/ruogu.fang/pliu1/BRATS21/main_FinetuningSwinUNETR_4Channels_Fewshot.py \
--json_list_train=$json_list_train  --json_list_test=$json_list_test \
--distributed --data_dir=$path_to_data_dir --val_every=10 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=$feature_size --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir