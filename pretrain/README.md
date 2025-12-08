# Self-Supervised Pretraining on UKBiobankhttps://github.com/lab-smile/BrainSegFounder/blob/main/pretrain/README.md
This directory contains scripts for creating the BrainSegFounder stage 1 pretraining model, and hosts scripts adapted from [here](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain).

## Data
The data used in our work for SSL pretraining is sourced from the [UK Biobank](https://www.ukbiobank.ac.uk/). For information on how to obtain access, please see their site. The UKB unique identifiers for pretraining images can be found in the [Google drive](https://drive.google.com/drive/folders/1fl3FeMEhv_cnIwrDa5geHPbKL-tHAuQE?usp=drive_link)

## Running
To run with only one modalitiy of images on a single GPU for 100 epochs, run:
```
python main.py --json "path/to/file.json" --epochs 100 --in_channels 1
```

To run with 2 modalities on multiple GPUs, run:
```
python main_T1T2.py --json "path/to/file.json" --distributed --epochs 300 --in_channels 2 --lr 5e-4
```
