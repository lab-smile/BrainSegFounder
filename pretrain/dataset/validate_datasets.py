import json

import os

import pathlib

def dataset_summary():
    workdir = pathlib.Path("/red/ruogu.fang/yyang/Pretrain")
    json_dir = workdir / 'jsons'
    print(json_dir)
    json_files = json_dir.glob("*.json")

    data_dir = workdir / "datasets"

    datasets_summary = open("datasets_summary.csv", 'w')
    datasets_summary.write("JSON, Train, Validation \n")
    for file_name in json_files:
        with open(file_name) as f: 
            #print(f.readlines())
            data = json.loads(f.read())
            data_info = f"{os.path.basename(file_name)}, {len(data['training'])}, {len(data['validation'])}"
            print(data_info)
            datasets_summary.write(data_info + "\n")

from monai.data import load_decathlon_datalist

def get_loader(args):
    splits1 = "/dataset_LUNA16_0.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_HNSCC_0.json"
    splits4 = "/dataset_TCIAcolon_UFL.json" #"/dataset_TCIAcolon_v2_0.json"
    splits5 = "/dataset_LIDC_0.json"
    list_dir = "./jsons"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    datadir1 = './dataset/dataset1' # "/dataset/dataset1"
    datadir2 = "./dataset/dataset2"
    datadir3 = "./dataset/dataset3"
    datadir4 = "./dataset/dataset4"
    datadir5 = "./dataset/dataset8"
    num_workers = 4
    
    if True:
        datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
        print("Dataset 1 LUNA16: number of data: {}".format(len(datalist1)))
        new_datalist1 = []
        for item in datalist1:
            item_dict = {"image": item["image"]}
            new_datalist1.append(item_dict)
        print(new_datalist1[-1])
        missed = check_if_exists(new_datalist1)
        print(f"missed {missed} files")
    if True:
        datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
        print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))
        print(datalist2[-1])
        missed = check_if_exists(datalist2)
        print(f"missed {missed} files")
    if True:
        datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
        print("Dataset 3 HNSCC: number of data: {}".format(len(datalist3)))
        print(datalist3[-1])
        missed = check_if_exists(datalist3)
        print(f"missed {missed} files")
    if True:
        datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
        print("Dataset 4 TCIA Colon: number of data: {}".format(len(datalist4)))
        print(datalist4[-1])
        missed = check_if_exists(datalist4)
        print(f"missed {missed} files")
    if True:
        datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
        print("Dataset 5: LIDC number of data: {}".format(len(datalist5)))
        print(datalist5[-1])
        missed = check_if_exists(datalist5)
        print(f"missed {missed} files")


def check_if_exists(datalist):
    counter = 0
    for item in datalist:
        file_path = item['image']
        if not os.path.isfile(file_path):
            #print(f"{file_path} not found")
            counter += 1
    return counter

import argparse
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--datadir", default="dataset", type=str, help="directory to save the dataset")

args = parser.parse_args()

get_loader(args)
