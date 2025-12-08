# BraTS finetuning
This directory contains the scripts for finetuning the BrainSegFounder model on BraTS.

## Requirements
To finetune on BraTS you will need:
1. BraTS data downloaded
2. The json file containing fold data, found on the Google Drive.
3. A pretrained model to finetune on.

## Running Finetuning
An example SLURM script for running finetuning can be found in [launch.sh](https://github.com/lab-smile/BrainSegFounder/blob/main/downstream/BraTS/finetuning/launch.sh), which calls main_FinetuningSwinUNETR_4Channels.py.

You can also run without singularity by directly calling the main_FinetuningSwinUNETR_4Channels.py python script with desired arguments.
