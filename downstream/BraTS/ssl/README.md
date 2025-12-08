# SSL (Stage 2) Pretraining on BraTS.

This directory hosts scripts adapted from [here](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain).  

## Requrements
To run Stage 2 pretraining on BraTS, you will need
1. The JSON file detailing folds for the data. This can be found on the google drive.
2. (Optionally) A stage 1 pretrained model. If you do not have one, pass no argument to the resume parameter.  

## Running

An example shell (slurm) script for can be found at launch.sh, which calls main_T1T2.py.
