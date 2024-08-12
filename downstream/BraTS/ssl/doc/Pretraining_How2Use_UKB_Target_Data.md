# How to use UKB and Target Data for Pretraining

There are two types of datasets for pretraining: a large scale of data (UKB data) and target data (e.g., BraTS data) 
for downstream tasks.

We use two stages of pretraining strategy. 

## Stage 1: Pretraining with UKB Data

The first stage is to pretrain the model with the UKB data, which has 40k subjects, and each subject has multiple modalities. 
In this work, we use the 2-channel data to do pretraining on UKB, which has T1 and T2 modalities for each subject.
### NOTE:
we may investigate the 4-channel data in the future, which has T1, T2, and other modalities for each subject. 



## Stage 2: Pretraining with Target Data (Self-pretraining)

The second stage is to use the pre-trained weights in Stage 1 to
further pretrain the model with the training part of the target data. 

We need to adapt the input channels of the pre-trained model in Stage 1 to the target data.  BRATS21 data has 4 modalities, 
i.e., T1, T1c, T2, and FLAIR. We change the network pre-trained in Stage 1 to have 4 channels 
to do pretraining on target data such as BRATS21.

### How to adapt the input channels of the pre-trained model to the target data

Two steps to adapt the input channels of the pre-trained model to the target data:
1. Change the input channels of the pre-trained model to 4 channels. In the code "main_T1T2_Stage2.py", we used the following code to change the input channels of the pre-trained model to 4 channels:
```
 if args.resume:

        print(f"[{args.rank}] " + f"Loading checkpoint from {args.pretrained_model_stage1}")
        # Access the PatchEmbed module within SwinViT
        patch_embed_layer = model.swinViT.patch_embed

        # Create a new convolutional layer with 4 input channels for 3D data
        new_proj = nn.Conv3d(4, patch_embed_layer.embed_dim, kernel_size=patch_embed_layer.patch_size,
                             stride=patch_embed_layer.patch_size)

        # Initialize the weights for the new channels
        with torch.no_grad():
            # Get the original weights
            original_weights = patch_embed_layer.proj.weight.clone()

            # Modify only the weights for the additional channels as needed
            # For example, re-initialize weights for channels 3 and 4
            nn.init.kaiming_normal_(original_weights[:, 2:4, :, :, :], mode='fan_out', nonlinearity='relu')

            # Assign the modified weights back to the layer
            patch_embed_layer.proj.weight = nn.Parameter(original_weights)

        # Replace the original proj layer with the new layer
        patch_embed_layer.proj = new_proj

        # Load the pre-trained model weights
        checkpoint = torch.load(args.pretrained_model_stage1)
        pretrained_state_dict = checkpoint['state_dict']

        # Prepare a new state dictionary for SwinUNETR's SwinViT part
        new_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith('module.swinViT.'):
                new_key = k.replace('module.swinViT.', '')  # Remove the prefix
                # Skip loading weights for the PatchEmbed proj layer
                if new_key != 'patch_embed.proj.weight' and new_key != 'patch_embed.proj.bias':
                    new_state_dict[new_key] = v

        # Load the pre-trained weights into SwinUNETR's SwinViT
        # Use strict=False to allow for the discrepancy in the first layer
        model.swinViT.load_state_dict(new_state_dict, strict=False)
        #put model to device
        model.to(args.device)

```
2. Add new dataloader for target data. In the code "utils/data_utils.py", we used the following code to add new dataloader for target data (BRATS21):

```
#read target data from dataset Brats
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def load_Brats_datalist(args):
    data_dir = args.target_data_path
    datalist_json = args.split_json
    fold = args.target_data_fold
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    training_images = train_files
    validation_images = validation_files

    return {'training': training_images,'validation': validation_images}

#add new dataloader function into the following code:
def get_T1T2_dataloaders(args,  num_workers = 4):

    if args.T1T2_10k:
        datalist = load_T1T210K_datalist(args)
    elif args.T1T2_10k_mixed:
        datalist = load_T1T210K_mixed_datalist(args)
    elif args.T1T2_40k_matched:
        datalist = load_T1T2matched_datalist(args)
    elif args.T1T2_target_Brats:
        print("Loading Brats dataset")
        datalist = load_Brats_datalist(args)
    else:
        raise ValueError("Unsupported dataset")
```
 
### How to run the code for pretraining with target data
Use the following code as en example, which is located at ./SLURM/launch_multi_2GPUs_-T1T2-modelS_Stage2.sh.
The code is to run the pretraining with target data (BRATS21) using the small size of the model (SwinUNETR-S)
pretrained on the first stage (UKB data).
```
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
#SBATCH --job-name=Stage2Pretrain_TargetBRATS_SmallT1T2
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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com


target_data_split_json="/mnt/jsons/brats21_folds.json"
target_data_path="/red/ruogu.fang/brats/"
learning_rate=6e-6
num_workers=4
BATCH_SIZE=2
TOTAL_EPOCHS=100
in_channels=4
pretrained_model_stage1="/red/ruogu.fang/yyang/SwinUNETR_pretrain_2channel/runs/run_T1T2_S_GPU064_D18_H3_07-03-2023-12:09:54-1535674/model_bestValRMSE.pt"
target_data_fold=0 #BRAST21 fold 0 is used for validation  ANOTHER 4 FOLDS ARE USED FOR TRAINING
NUM_STEPS=100000
EVAL_NUM=500
num_swin_block=18
num_heads_first_stage=3
bottleneck_depth=768
feature_size=48
spatial_dims=3

export CONTAINER_NAME=/blue/ruogu.fang/pliu1/GatorBrainContainer

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
--noamp \
--split /mnt/jsons/GBR_T1T2_matched_image.json \
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

```
To run it: 
```
cd SWINUNETR_pretrain_2channel
sbatch ./SLURM/launch_multi_2GPUs_-T1T2-modelS_Stage2.sh

```
# BRATS21 Data


BRATS21 has 1,251 + 219 = 1,470 subjects. 

There are 1,251 located in /red/ruogu.fang/brats/TrainingData
There are 219 located in /red/ruogu.fang/brats/ValidationData

The TrainingData was used for training and validation, and the ValidationData was used for testing.


The TrainingData was split into 5 folds.
During training, 4 folds were used for training and 1 fold was used for validation.


See the json file located in jsons/ for the split details.
