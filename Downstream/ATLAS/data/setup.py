"""Code to set up the current directory for the ATLAS v2.0 dataset.
This script must be run *before* . In addition, it assumes you have the encryption key for the ATLAS dataset available,
instructions for receiving this key can be found at https://fcon_1000.projects.nitrc.org/indi/retro/atlas_download.html

Much of the code in this directory is sourced directly from the ATLAS 2.0 GitHub's tutorial page.
"""
import wget
import pathlib
import json
import shutil
import os
from os.path import join
import hashlib
import getpass
import subprocess


def data_fetch(check_hash=True) -> None:
    wget.download(data["url"])
    filename = os.path.basename(data["url"])

    if check_hash:
        print("")
        print("Checking data integrity; this may take a few minutes.")
        if check_hash_correct(filename, data["encrypted_hash"]):
            print("Data verified to be correct.")
        else:
            print(
                "There is something wrong with the data. Verify that the expected files are present."
            )
    return


def check_hash_correct(filename: str, expected_hash: str) -> bool:
    return get_sha256(filename) == expected_hash


def get_sha256(filename: str, block_size: int = 2 ** 16) -> str:
    sha256 = hashlib.sha256()
    f = open(filename, "rb")
    file_data = f.read(block_size)
    while len(file_data) > 0:
        sha256.update(file_data)
        file_data = f.read(block_size)
    f.close()
    return sha256.hexdigest()


def bidsify_indi_atlas(atlas_path: str, destination_path: str = 'data') -> None:
    training_source = join(atlas_path, "Training")
    testing_source = join(atlas_path, "Testing")

    # Create destination if needed
    dest = pathlib.Path(destination_path)
    training_dest = pathlib.Path(dest).joinpath("train")
    derivatives_dest = training_dest.joinpath(
        "derivatives", training_config["data_derivatives_names"][0]
    )

    testing_dest = pathlib.Path(dest).joinpath("test")
    derivatives_test_dest = testing_dest.joinpath(
        "derivatives", testing_config["data_derivatives_names"][0]
    )

    derivatives_dest.mkdir(parents=True, exist_ok=True)
    derivatives_test_dest.mkdir(parents=True, exist_ok=True)

    # Data is in ATLAS_2/Training/Rxxx/
    # Move out of Rxxx; dataset_description.json is the same across all subjects, so we can just ignore it.
    # If we're on the same filesystem, we can just move the files.
    dev_source = os.stat(atlas_path).st_dev
    dev_dest = os.stat(destination_path).st_dev
    same_fs = dev_source == dev_dest

    if same_fs:
        move_file = os.rename
        move_dir = os.rename
    else:
        move_file = shutil.copy2
        move_dir = shutil.copytree

    # Move files over!
    _merge_cohort_data(training_source, derivatives_dest, move_dir, move_file)
    _merge_cohort_data(testing_source, derivatives_test_dest, move_dir, move_file)

    # Write dataset_description.json to top-level training dir
    dataset_desc = {"Name": "ATLAS", "BIDSVersion": "1.6.0", "Authors": ["NPNL"]}
    dataset_desc_path = training_dest.joinpath("dataset_description.json")
    f = open(dataset_desc_path, "w")
    json.dump(dataset_desc, f, separators=(",\n", ":\t"))
    f.close()

    dataset_desc_test_path = testing_dest.joinpath("dataset_description.json")
    f = open(dataset_desc_test_path, "w")
    json.dump(dataset_desc, f, separators=(",\n", ":\t"))
    f.close()
    return


def _merge_cohort_data(root_dir: str, derivatives_dest: str, move_dir_func: callable, move_file_func: callable) -> None:
    dataset_description_path = ''
    for r_dir in os.listdir(root_dir):
        if r_dir.startswith("."):
            continue  # There are hidden files spread out; we don't need them.
        leading_path = join(root_dir, r_dir)
        for sub in os.listdir(leading_path):
            if sub.startswith("."):
                continue  # As above
            path_to_move = join(leading_path, sub)
            destination = join(derivatives_dest, sub)
            if sub == "dataset_description.json":
                dataset_description_path = path_to_move
                continue
            if pathlib.Path(path_to_move).is_dir():
                move_dir_func(path_to_move, destination)
            else:
                move_file_func(path_to_move, destination)
    shutil.copy2(dataset_description_path, derivatives_dest.joinpath("dataset_description.json"))
    return


if __name__ == '__main__':
    # CONSTANTS
    data_path = '..'

    data = {
        "encrypted_hash": "b9cdf26486e7dd325d5d6617f2218204bbaa0b649dbca03e729a41a449bef671",
        "url": "ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/ATLAS/releases/R2.0/ATLAS_R2.0_encrypted.tar.gz",
        "private_osf_ids": ["2rvym", "3t8jg", "nkr2e"],
    }

    training_config = {
        "batch_size": 5,
        "dir_name": join(data_path, "train"),
        "data_entities": [{"subject": "", "session": "", "suffix": "T1w"}],
        "target_entities": [{"label": "L", "desc": "T1lesion", "suffix": "mask"}],
        "data_derivatives_names": ["ATLAS"],
        "target_derivatives_names": ["ATLAS"],
        "label_names": ["not lesion", "lesion"],
    }

    cross_validation = {"n_splits": 5, "train_size": 0.6, "random_state": 9001}

    testing_config = {
        "dir_name": join(data_path, "test"),
        "batch_size": training_config["batch_size"],
        "test_dir_name": "test",
        "data_entities": [{"subject": "", "session": "", "suffix": "T1w"}],
        "target_entities": [{"label": "L", "desc": "T1lesion", "suffix": "mask"}],
        "data_derivatives_names": ["ATLAS"],
        "target_derivatives_names": ["ATLAS"],
        "label_names": ["not lesion", "lesion"],
    }

    data_fetch(check_hash=True)

    subprocess.call(['openssl', 'aes-256-cbc', '-md', 'sha256',
                     '-d', '-a', '-in',
                     'ATLAS_R2.0_encrypted.tar.gz', '-out', 'ATLAS_R2.0.tar.gz',
                     '-pass', f'pass:{getpass.getpass("Enter password")}'])

    subprocess.call(['tar', '-xzf', 'ATLAS_R2.0.tar.gz'])

    # Make compatible with PyBIDS:
    bidsify_indi_atlas(atlas_path='ATLAS_2/', destination_path='/')
