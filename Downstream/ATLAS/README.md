# ATLAS v2.0 Dataset

This directory contains the machine learning pipeline for training models on the ATLAS v2.0 dataset. 
The pipeline includes scripts for downloading the dataset and setting up a workspace, training models, and download 
links for our pretrained models.

## Prerequisites

These models were trained using Project MONAI's singularity container (version 1.3.0). We recommend creating such a 
container on your own machine. This code was written and tested on Python 3.10. The requirements for training are 
Project MONAI's [dev requirements](https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt) 
and the NPNL Lab's [BIDSIO library](https://github.com/npnl/bidsio/tree/main).

You can install both with pip using the following commands:

```bash
pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt
pip install -U git+https://github.com/npnl/bidsio
```

## Dataset setup

To set up a directory for training and finetuning, ensure you have: 
1. A Stage 1 pretrained model wit

