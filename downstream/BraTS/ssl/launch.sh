#!/bin/bash
#
# Script to launch a multi-gpu distributed training using MONAI Core
# on UF HiperGator's AI partition, a SLURM cluster using Singularity 
# as container runtime.
# 
# This script uses `pt_multinode_helper_funcs.sh`, and 
# either `run_on_node.sh`(for single-node multi-gpu training) 
# or `run_on_multinode.sh` (for multi-node multi-gpu training). All
# the three `.sh` files are in \monaicore_multigpu\util_multigpu.
#
# We use torch.distributed.launch to launch the training, so please 
# set as follows: 
#   set #SBATCH --ntasks=--nodes
#   set #SBATCH --ntasks-per-node=1  
#   set #SBATCH --gpus=total number of processes to run on all nodes
#   set #SBATCH --gpus-per-task=--gpus/--ntasks  
#
#   for multi-node training, replace `run_on_node.sh` in 
#   `PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")`
#   with `run_on_multinode.sh`.
#   
#   Modify paths to your own paths.
#      
# (c) 2021, Brian J. Stucky, UF Research Computing
# 2022, modified by Huiwen Ju, hju@nvidia.com

# Resource allocation.
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=ExampleJob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --partition=hpg-ai
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out

learning_rate=6e-6 #learning rate needs to be increased when using more GPUs
num_workers=4
BATCH_SIZE=2
NUM_STEPS=50000
EVAL_NUM=500
TOTAL_EPOCHS=100

#data
target_data_split_json="/path/to/json/file.json"
target_data_path="/path/to/brats/"
target_data_fold=0 #BRAST21 fold 0 is used for validation  ANOTHER 4 FOLDS ARE USED FOR TRAINING

#structure of the model
pretrained_model_stage1="/path/to/pretrained/model"
in_channels=4
num_swin_block=2
num_heads_first_stage=3
bottleneck_depth=768
feature_size=48
spatial_dims=3

export CONTAINER_NAME=ContainerName

# Training command specification: training_script -args.
TRAINING_CMD="/mnt/main_T1T2_Stage2.py \
--resume \
--distributed \
--logdir=/mnt/runs \
--workdir=/mnt \
--roi_x=96 --roi_y=96 --roi_z=96 \
--lrdecay --lr=$learning_rate --decay=0.1 \
--batch_size=$BATCH_SIZE \
--epochs=$TOTAL_EPOCHS \
--num_steps=$NUM_STEPS \
--eval_num=$EVAL_NUM \
--in_channels=$in_channels \
--num_swin_block=$num_swin_block \
--num_heads_first_stage=$num_heads_first_stage \
--bottleneck_depth=$bottleneck_depth \
--num_workers=$num_workers \
--feature_size=$feature_size --spatial_dims=$spatial_dims \
--target_data_fold=$target_data_fold \
--pretrained_model_stage1=$pretrained_model_stage1 \
--split_json=$target_data_split_json \
--target_data_path=$target_data_path \
--T1T2_target_Brats --noamp \
"


PYTHON_PATH="singularity exec --nv --bind /red,$(pwd):/mnt \
          $CONTAINER_NAME python3"


export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1


# Location of the PyTorch launch utilities, 
# i.e. `pt_multinode_helper_funcs.sh`, `run_on_node.sh` and `run_on_multinode`.
PT_LAUNCH_UTILS_PATH=$(pwd)/util_multigpu
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

#PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")
PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_multinode.sh")
echo "Running \"$TRAINING_CMD\" on each node..."

srun --unbuffered "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"   "$PRIMARY" "$PRIMARY_PORT"

