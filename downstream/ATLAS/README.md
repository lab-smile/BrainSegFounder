# ATLAS v2.0 Dataset

This directory contains the machine learning pipeline for training models on the ATLAS v2.0 dataset. 
The pipeline includes scripts for downloading the dataset and setting up a workspace, training models, and download 
links for our pretrained models.

In addition, we have provided example SLURM scripts that were used to train the models used for download.

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

To set up a directory with data, run `python data/setup.py`. This will require you to have a decryption key for the ATLAS
dataset, which you can obtain from the ATLAS website directly after application for the data. 

## Stage 2 Pretraining
To pretrain a model for Stage 2 on ATLAS with a batch size of 1 learning rate decay on multi-gpu, you can run the following command: 
```bash
python python pretrain.py -b 1 -d data/ -p pretrained.pt -w 4 --distributed --lr_decay --max_grad_norm 5.0 --amp --url "tcp://127.0.0.1:24734"
```

To pretrain a model from scratch (without a pretrained Stage 1 model), with a batch size of 3 for 5000 epochs, you can run
```bash
python python pretrain.py -b 3 -e 5000 -d data/ -w 4 --distributed --lr_decay --max_grad_norm 5.0 --amp --url "tcp://127.0.0.1:24734"
```

## Finetuning
To finetune a pretrained model on ATLAS, you can run:

```bash
python finetune.py --checkpoint finetuned_model.pt --logdir logs/finetune/ --data_dir data/ --output models/ --num_workers 1 --batch_size 2 --epochs 1000 --seed 1234 --distributed --in_channels 1 --out_channels 1 --feature_size 48 --depths 2 2 2 2 --dropout_rate 0.1 --amp --heads 3 6 12 24
```

## Model Downloads

Pretrained models for the ATLAS dataset can be downloaded [here](https://drive.google.com/drive/folders/1fl3FeMEhv_cnIwrDa5geHPbKL-tHAuQE?usp=drive_link)
